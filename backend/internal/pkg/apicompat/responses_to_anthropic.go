package apicompat

import (
	"encoding/json"
	"fmt"
	"time"
)

// ---------------------------------------------------------------------------
// Non-streaming: ResponsesResponse → AnthropicResponse
// ---------------------------------------------------------------------------

// ResponsesToAnthropic converts a Responses API response directly into an
// Anthropic Messages response. Reasoning output items are mapped to thinking
// blocks; function_call items become tool_use blocks.
func ResponsesToAnthropic(resp *ResponsesResponse, model string) *AnthropicResponse {
	out := &AnthropicResponse{
		ID:    resp.ID,
		Type:  "message",
		Role:  "assistant",
		Model: model,
	}
	if resp.Conversation != nil {
		out.Container = responsesConversationToAnthropicContainer(resp.Conversation)
	}

	var blocks []AnthropicContentBlock

	for _, item := range resp.Output {
		switch item.Type {
		case "reasoning":
			summaryText := ""
			for _, s := range item.Summary {
				if s.Type == "summary_text" && s.Text != "" {
					summaryText += s.Text
				}
			}
			if summaryText == "" {
				summaryText = thinkingPlaceholder
			}
			sig := encodeReasoningItemSignature(item)
			blocks = append(blocks, AnthropicContentBlock{
				Type:      "thinking",
				Thinking:  summaryText,
				Signature: sig,
			})
		case "compaction":
			if item.ID != "" && item.EncryptedContent != "" {
				blocks = append(blocks, AnthropicContentBlock{
					Type:      "thinking",
					Thinking:  thinkingPlaceholder,
					Signature: encodeCompactionItemSignature(item),
				})
			}
		case "message":
			for _, part := range item.Content {
				switch part.Type {
				case "output_text":
					if part.Text != "" {
						blocks = append(blocks, AnthropicContentBlock{
							Type: "text",
							Text: part.Text,
						})
					}
				case "refusal":
					if part.Refusal != "" {
						blocks = append(blocks, AnthropicContentBlock{
							Type: "text",
							Text: part.Refusal,
						})
					}
				}
			}
		case "function_call":
			blockType := "tool_use"
			if isResponsesMCPNamespace(item.Namespace, item.RawItem) {
				blockType = "mcp_tool_use"
			}
			blocks = append(blocks, AnthropicContentBlock{
				Type:       blockType,
				ID:         fromResponsesCallIDToAnthropic(item.CallID),
				Name:       item.Name,
				ServerName: item.Namespace,
				Input:      sanitizeAnthropicToolUseInput(item.Name, item.Arguments),
			})
		case "web_search_call":
			toolUseID := "srvtoolu_" + item.ID
			query := ""
			if item.Action != nil {
				query = item.Action.Query
			}
			inputJSON, _ := json.Marshal(map[string]string{"query": query})
			blocks = append(blocks, AnthropicContentBlock{
				Type:  "server_tool_use",
				ID:    toolUseID,
				Name:  "web_search",
				Input: inputJSON,
			})
			emptyResults, _ := json.Marshal([]struct{}{})
			blocks = append(blocks, AnthropicContentBlock{
				Type:      "web_search_tool_result",
				ToolUseID: toolUseID,
				Content:   emptyResults,
			})
		}
	}

	if len(blocks) == 0 {
		blocks = append(blocks, AnthropicContentBlock{Type: "text", Text: ""})
	}
	out.Content = blocks

	out.StopReason = responsesStatusToAnthropicStopReason(resp.Status, resp.IncompleteDetails, blocks)

	if resp.Usage != nil {
		out.Usage = anthropicUsageFromResponsesUsage(resp.Usage)
	}

	return out
}

func responsesConversationToAnthropicContainer(conversation *ResponsesConversation) json.RawMessage {
	if conversation == nil || conversation.ID == "" {
		return nil
	}
	payload, err := json.Marshal(conversation)
	if err != nil {
		return nil
	}
	return payload
}

func anthropicUsageFromResponsesUsage(usage *ResponsesUsage) AnthropicUsage {
	if usage == nil {
		return AnthropicUsage{}
	}

	cachedTokens := 0
	if usage.InputTokensDetails != nil {
		cachedTokens = usage.InputTokensDetails.CachedTokens
	}

	inputTokens := usage.InputTokens - cachedTokens
	if inputTokens < 0 {
		inputTokens = 0
	}

	return AnthropicUsage{
		InputTokens:          inputTokens,
		OutputTokens:         usage.OutputTokens,
		CacheReadInputTokens: cachedTokens,
	}
}

func responsesStatusToAnthropicStopReason(status string, details *ResponsesIncompleteDetails, blocks []AnthropicContentBlock) string {
	switch status {
	case "incomplete":
		if details != nil && details.Reason == "max_output_tokens" {
			return "max_tokens"
		}
		return "end_turn"
	case "completed":
		if len(blocks) > 0 && (blocks[len(blocks)-1].Type == "tool_use" || blocks[len(blocks)-1].Type == "mcp_tool_use") {
			return "tool_use"
		}
		return "end_turn"
	default:
		return "end_turn"
	}
}

func sanitizeAnthropicToolUseInput(name string, raw string) json.RawMessage {
	if name != "Read" || raw == "" {
		return json.RawMessage(raw)
	}

	var input map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &input); err != nil {
		return json.RawMessage(raw)
	}

	if pages, ok := input["pages"]; !ok || string(pages) != `""` {
		return json.RawMessage(raw)
	}

	delete(input, "pages")
	sanitized, err := json.Marshal(input)
	if err != nil {
		return json.RawMessage(raw)
	}
	return sanitized
}

// ---------------------------------------------------------------------------
// Streaming: ResponsesStreamEvent → []AnthropicStreamEvent (stateful converter)
// ---------------------------------------------------------------------------

// ResponsesEventToAnthropicState tracks state for converting a sequence of
// Responses SSE events directly into Anthropic SSE events.
type ResponsesEventToAnthropicState struct {
	MessageStartSent bool
	MessageStopSent  bool

	ContentBlockIndex int
	ContentBlockOpen  bool
	CurrentBlockType  string // "text" | "thinking" | "tool_use"
	CurrentToolName   string
	CurrentToolArgs   string

	HasReasoningDelta bool

	// BlockHasDelta tracks whether each block index has received any delta.
	BlockHasDelta map[int]bool

	// OutputIndexToBlockIdx maps Responses output_index → Anthropic content block index.
	OutputIndexToBlockIdx map[int]int

	// FuncCallWhitespaceCount tracks consecutive whitespace chars per output_index.
	FuncCallWhitespaceCount map[int]int

	InputTokens          int
	OutputTokens         int
	CacheReadInputTokens int

	ResponseID string
	Model      string
	Created    int64
}

// NewResponsesEventToAnthropicState returns an initialised stream state.
func NewResponsesEventToAnthropicState() *ResponsesEventToAnthropicState {
	return &ResponsesEventToAnthropicState{
		OutputIndexToBlockIdx:   make(map[int]int),
		FuncCallWhitespaceCount: make(map[int]int),
		BlockHasDelta:           make(map[int]bool),
		Created:                 time.Now().Unix(),
	}
}

// ResponsesEventToAnthropicEvents converts a single Responses SSE event into
// zero or more Anthropic SSE events, updating state as it goes.
func ResponsesEventToAnthropicEvents(
	evt *ResponsesStreamEvent,
	state *ResponsesEventToAnthropicState,
) []AnthropicStreamEvent {
	switch evt.Type {
	case "response.created":
		return resToAnthHandleCreated(evt, state)
	case "response.output_item.added":
		return resToAnthHandleOutputItemAdded(evt, state)
	case "response.content_part.added":
		return resToAnthHandleContentPartAdded(evt, state)
	case "response.content_part.done":
		return resToAnthHandleContentPartDone(evt, state)
	case "response.output_text.delta":
		return resToAnthHandleTextDelta(evt, state)
	case "response.output_text.done":
		return resToAnthHandleOutputTextDone(evt, state)
	case "response.refusal.delta":
		return resToAnthHandleRefusalDelta(evt, state)
	case "response.refusal.done":
		return resToAnthHandleRefusalDone(evt, state)
	case "response.function_call_arguments.delta":
		return resToAnthHandleFuncArgsDelta(evt, state)
	case "response.function_call_arguments.done":
		return resToAnthHandleFuncArgsDone(evt, state)
	case "response.output_item.done":
		return resToAnthHandleOutputItemDone(evt, state)
	case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
		return resToAnthHandleReasoningDelta(evt, state)
	case "response.reasoning_text.done", "response.reasoning_summary_text.done":
		return resToAnthHandleReasoningSummaryDone(evt, state)
	case "response.completed", "response.incomplete":
		return resToAnthHandleCompleted(evt, state)
	case "response.failed":
		return resToAnthHandleFailed(evt, state)
	case "error":
		return resToAnthHandleError(evt, state)
	default:
		return nil
	}
}

// FinalizeResponsesAnthropicStream emits synthetic termination events if the
// stream ended without a proper completion event.
func FinalizeResponsesAnthropicStream(state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	if !state.MessageStartSent || state.MessageStopSent {
		return nil
	}

	var events []AnthropicStreamEvent
	events = append(events, closeCurrentBlock(state)...)

	events = append(events,
		AnthropicStreamEvent{
			Type: "message_delta",
			Delta: &AnthropicDelta{
				StopReason: "end_turn",
			},
			Usage: &AnthropicUsage{
				InputTokens:          state.InputTokens,
				OutputTokens:         state.OutputTokens,
				CacheReadInputTokens: state.CacheReadInputTokens,
			},
		},
		AnthropicStreamEvent{Type: "message_stop"},
	)
	state.MessageStopSent = true
	return events
}

// ResponsesAnthropicEventToSSE formats an AnthropicStreamEvent as an SSE line pair.
func ResponsesAnthropicEventToSSE(evt AnthropicStreamEvent) (string, error) {
	data, err := json.Marshal(evt)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("event: %s\ndata: %s\n\n", evt.Type, data), nil
}

// --- internal handlers ---

func resToAnthHandleCreated(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	if evt.Response != nil {
		state.ResponseID = evt.Response.ID
		// Only use upstream model if no override was set (e.g. originalModel)
		if state.Model == "" {
			state.Model = evt.Response.Model
		}
	}

	if state.MessageStartSent {
		return nil
	}
	state.MessageStartSent = true

	return []AnthropicStreamEvent{{
		Type: "message_start",
		Message: &AnthropicResponse{
			ID:      state.ResponseID,
			Type:    "message",
			Role:    "assistant",
			Content: []AnthropicContentBlock{},
			Model:   state.Model,
			Usage: AnthropicUsage{
				InputTokens:  0,
				OutputTokens: 0,
			},
		},
	}}
}

func resToAnthHandleOutputItemAdded(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	if evt.Item == nil {
		return nil
	}

	switch evt.Item.Type {
	case "function_call":
		var events []AnthropicStreamEvent
		events = append(events, closeCurrentBlock(state)...)

		idx := state.ContentBlockIndex
		state.OutputIndexToBlockIdx[evt.OutputIndex] = idx
		state.ContentBlockOpen = true
		state.CurrentBlockType = "tool_use"
		state.CurrentToolName = evt.Item.Name
		state.CurrentToolArgs = ""

		blockType := "tool_use"
		if isResponsesMCPNamespace(evt.Item.Namespace, evt.Item.RawItem) {
			blockType = "mcp_tool_use"
		}
		contentBlock := &AnthropicContentBlock{
			Type:       blockType,
			ID:         evt.Item.CallID,
			Name:       evt.Item.Name,
			ServerName: evt.Item.Namespace,
			Input:      json.RawMessage("{}"),
		}

		events = append(events, AnthropicStreamEvent{
			Type:         "content_block_start",
			Index:        &idx,
			ContentBlock: contentBlock,
		})
		return events

	case "reasoning":
		var events []AnthropicStreamEvent
		events = append(events, closeCurrentBlock(state)...)

		idx := state.ContentBlockIndex
		state.OutputIndexToBlockIdx[evt.OutputIndex] = idx
		state.ContentBlockOpen = true
		state.CurrentBlockType = "thinking"

		events = append(events, AnthropicStreamEvent{
			Type:  "content_block_start",
			Index: &idx,
			ContentBlock: &AnthropicContentBlock{
				Type:     "thinking",
				Thinking: "",
			},
		})
		return events

	case "message":
		return nil
	}

	return nil
}

func resToAnthHandleContentPartAdded(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	part := decodeResponsesOutputPart(evt.Part)
	if part == nil {
		return nil
	}

	switch part.Type {
	case "output_text":
		if part.Text == "" {
			return nil
		}
		clone := *evt
		clone.Delta = part.Text
		return resToAnthHandleTextDelta(&clone, state)
	case "refusal":
		text := part.Refusal
		if text == "" {
			text = part.Text
		}
		if text == "" {
			return nil
		}
		clone := *evt
		clone.Delta = text
		clone.Refusal = text
		return resToAnthHandleRefusalDelta(&clone, state)
	default:
		return nil
	}
}

func resToAnthHandleContentPartDone(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	part := decodeResponsesOutputPart(evt.Part)
	if part == nil {
		return nil
	}

	switch part.Type {
	case "output_text":
		clone := *evt
		clone.Text = part.Text
		return resToAnthHandleOutputTextDone(&clone, state)
	case "refusal":
		text := part.Refusal
		if text == "" {
			text = part.Text
		}
		clone := *evt
		clone.Refusal = text
		clone.Text = text
		return resToAnthHandleRefusalDone(&clone, state)
	default:
		return nil
	}
}

func resToAnthHandleRefusalDelta(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	text := evt.Delta
	if text == "" {
		text = evt.Refusal
	}
	if text == "" {
		return nil
	}
	clone := *evt
	clone.Delta = text
	return resToAnthHandleTextDelta(&clone, state)
}

func resToAnthHandleRefusalDone(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	text := evt.Refusal
	if text == "" {
		text = evt.Text
	}
	clone := *evt
	clone.Text = text
	return resToAnthHandleOutputTextDone(&clone, state)
}

func decodeResponsesOutputPart(raw json.RawMessage) *ResponsesOutputPart {
	if len(raw) == 0 {
		return nil
	}
	var part ResponsesOutputPart
	if err := json.Unmarshal(raw, &part); err == nil && part.Type != "" {
		return &part
	}
	var contentPart ResponsesContentPart
	if err := json.Unmarshal(raw, &contentPart); err == nil && contentPart.Type != "" {
		return &ResponsesOutputPart{
			Type:        contentPart.Type,
			Text:        contentPart.Text,
			Refusal:     contentPart.Refusal,
			Annotations: contentPart.Annotations,
			RawPart:     raw,
		}
	}
	return nil
}

func resToAnthHandleTextDelta(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	if evt.Delta == "" {
		return nil
	}

	var events []AnthropicStreamEvent

	if !state.ContentBlockOpen || state.CurrentBlockType != "text" {
		events = append(events, closeCurrentBlock(state)...)

		idx := state.ContentBlockIndex
		state.ContentBlockOpen = true
		state.CurrentBlockType = "text"

		events = append(events, AnthropicStreamEvent{
			Type:  "content_block_start",
			Index: &idx,
			ContentBlock: &AnthropicContentBlock{
				Type: "text",
				Text: "",
			},
		})
	}

	idx := state.ContentBlockIndex
	state.BlockHasDelta[idx] = true
	events = append(events, AnthropicStreamEvent{
		Type:  "content_block_delta",
		Index: &idx,
		Delta: &AnthropicDelta{
			Type: "text_delta",
			Text: evt.Delta,
		},
	})
	return events
}

const maxConsecutiveFuncCallWhitespace = 20

func resToAnthHandleFuncArgsDelta(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	if evt.Delta == "" {
		return nil
	}

	if state.CurrentBlockType == "tool_use" && state.CurrentToolName == "Read" {
		state.CurrentToolArgs += evt.Delta
		return nil
	}

	blockIdx, ok := state.OutputIndexToBlockIdx[evt.OutputIndex]
	if !ok {
		return nil
	}

	// Check for runaway whitespace in function call arguments
	count := state.FuncCallWhitespaceCount[evt.OutputIndex]
	for _, ch := range evt.Delta {
		if ch == '\r' || ch == '\n' || ch == '\t' {
			count++
			if count > maxConsecutiveFuncCallWhitespace {
				var events []AnthropicStreamEvent
				events = append(events, closeCurrentBlock(state)...)
				events = append(events, AnthropicStreamEvent{
					Type: "error",
					Error: &AnthropicErrorDetail{
						Type:    "api_error",
						Message: "Received function call arguments delta containing more than 20 consecutive whitespace characters.",
					},
				})
				state.MessageStopSent = true
				return events
			}
		} else if ch != ' ' {
			count = 0
		}
	}
	state.FuncCallWhitespaceCount[evt.OutputIndex] = count

	state.BlockHasDelta[blockIdx] = true
	return []AnthropicStreamEvent{{
		Type:  "content_block_delta",
		Index: &blockIdx,
		Delta: &AnthropicDelta{
			Type:        "input_json_delta",
			PartialJSON: evt.Delta,
		},
	}}
}

func resToAnthHandleReasoningDelta(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	if evt.Delta == "" {
		return nil
	}

	blockIdx, ok := state.OutputIndexToBlockIdx[evt.OutputIndex]
	if !ok {
		return nil
	}

	state.HasReasoningDelta = true
	state.BlockHasDelta[blockIdx] = true

	return []AnthropicStreamEvent{{
		Type:  "content_block_delta",
		Index: &blockIdx,
		Delta: &AnthropicDelta{
			Type:     "thinking_delta",
			Thinking: evt.Delta,
		},
	}}
}

// resToAnthHandleReasoningSummaryDone handles response.reasoning_summary_text.done.
// Per the reference implementation, this does NOT close the block — it only emits
// a fallback thinking_delta if no delta was previously sent for this block.
func resToAnthHandleReasoningSummaryDone(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	blockIdx, ok := state.OutputIndexToBlockIdx[evt.OutputIndex]
	if !ok {
		// Open a thinking block if needed
		var events []AnthropicStreamEvent
		events = append(events, closeCurrentBlock(state)...)
		blockIdx = state.ContentBlockIndex
		state.OutputIndexToBlockIdx[evt.OutputIndex] = blockIdx
		state.ContentBlockOpen = true
		state.CurrentBlockType = "thinking"
		events = append(events, AnthropicStreamEvent{
			Type:  "content_block_start",
			Index: &blockIdx,
			ContentBlock: &AnthropicContentBlock{
				Type:     "thinking",
				Thinking: "",
			},
		})
		if evt.Text != "" && !state.BlockHasDelta[blockIdx] {
			events = append(events, AnthropicStreamEvent{
				Type:  "content_block_delta",
				Index: &blockIdx,
				Delta: &AnthropicDelta{
					Type:     "thinking_delta",
					Thinking: evt.Text,
				},
			})
			state.BlockHasDelta[blockIdx] = true
		}
		return events
	}

	if evt.Text != "" && !state.BlockHasDelta[blockIdx] {
		state.BlockHasDelta[blockIdx] = true
		return []AnthropicStreamEvent{{
			Type:  "content_block_delta",
			Index: &blockIdx,
			Delta: &AnthropicDelta{
				Type:     "thinking_delta",
				Thinking: evt.Text,
			},
		}}
	}
	return nil
}

// resToAnthHandleOutputTextDone handles response.output_text.done.
// Does NOT close the block — only emits a fallback text_delta if no delta was sent.
func resToAnthHandleOutputTextDone(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	blockIdx := state.ContentBlockIndex
	if evt.Text != "" && !state.BlockHasDelta[blockIdx] {
		// Open text block if needed
		var events []AnthropicStreamEvent
		if !state.ContentBlockOpen || state.CurrentBlockType != "text" {
			events = append(events, closeCurrentBlock(state)...)
			blockIdx = state.ContentBlockIndex
			state.ContentBlockOpen = true
			state.CurrentBlockType = "text"
			events = append(events, AnthropicStreamEvent{
				Type:  "content_block_start",
				Index: &blockIdx,
				ContentBlock: &AnthropicContentBlock{
					Type: "text",
					Text: "",
				},
			})
		}
		events = append(events, AnthropicStreamEvent{
			Type:  "content_block_delta",
			Index: &blockIdx,
			Delta: &AnthropicDelta{
				Type: "text_delta",
				Text: evt.Text,
			},
		})
		state.BlockHasDelta[blockIdx] = true
		return events
	}
	return nil
}

// resToAnthHandleFuncArgsDone handles response.function_call_arguments.done.
// Does NOT close the block — only emits a fallback input_json_delta if no delta was sent.
func resToAnthHandleFuncArgsDone(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	blockIdx, ok := state.OutputIndexToBlockIdx[evt.OutputIndex]
	if !ok {
		return nil
	}
	if evt.Arguments != "" && !state.BlockHasDelta[blockIdx] {
		state.BlockHasDelta[blockIdx] = true
		return []AnthropicStreamEvent{{
			Type:  "content_block_delta",
			Index: &blockIdx,
			Delta: &AnthropicDelta{
				Type:        "input_json_delta",
				PartialJSON: evt.Arguments,
			},
		}}
	}
	return nil
}

func resToAnthHandleOutputItemDone(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	if evt.Item == nil {
		return nil
	}

	switch evt.Item.Type {
	case "web_search_call":
		if evt.Item.Status == "completed" {
			return resToAnthHandleWebSearchDone(evt, state)
		}
	case "reasoning":
		return resToAnthHandleReasoningItemDone(evt, state)
	case "compaction":
		return resToAnthHandleCompactionItemDone(evt, state)
	}

	if state.ContentBlockOpen {
		return closeCurrentBlock(state)
	}
	return nil
}

func resToAnthHandleReasoningItemDone(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	var events []AnthropicStreamEvent

	blockIdx, ok := state.OutputIndexToBlockIdx[evt.OutputIndex]
	if !ok {
		// Open a thinking block if one wasn't opened by reasoning_summary_text.delta
		events = append(events, closeCurrentBlock(state)...)
		blockIdx = state.ContentBlockIndex
		state.OutputIndexToBlockIdx[evt.OutputIndex] = blockIdx
		state.ContentBlockOpen = true
		state.CurrentBlockType = "thinking"
		events = append(events, AnthropicStreamEvent{
			Type:  "content_block_start",
			Index: &blockIdx,
			ContentBlock: &AnthropicContentBlock{
				Type:     "thinking",
				Thinking: "",
			},
		})
	}

	// If no summary deltas were sent, emit a placeholder thinking_delta
	if !state.HasReasoningDelta {
		events = append(events, AnthropicStreamEvent{
			Type:  "content_block_delta",
			Index: &blockIdx,
			Delta: &AnthropicDelta{
				Type:     "thinking_delta",
				Thinking: thinkingPlaceholder,
			},
		})
	}

	sig := encodeReasoningItemSignature(*evt.Item)
	events = append(events, AnthropicStreamEvent{
		Type:  "content_block_delta",
		Index: &blockIdx,
		Delta: &AnthropicDelta{
			Type:      "signature_delta",
			Signature: sig,
		},
	})

	state.HasReasoningDelta = false
	return events
}

func resToAnthHandleCompactionItemDone(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	if evt.Item.ID == "" || evt.Item.EncryptedContent == "" {
		return nil
	}

	var events []AnthropicStreamEvent
	events = append(events, closeCurrentBlock(state)...)

	idx := state.ContentBlockIndex
	state.ContentBlockOpen = true
	state.CurrentBlockType = "thinking"

	events = append(events, AnthropicStreamEvent{
		Type:  "content_block_start",
		Index: &idx,
		ContentBlock: &AnthropicContentBlock{
			Type:     "thinking",
			Thinking: "",
		},
	})

	events = append(events, AnthropicStreamEvent{
		Type:  "content_block_delta",
		Index: &idx,
		Delta: &AnthropicDelta{
			Type:     "thinking_delta",
			Thinking: thinkingPlaceholder,
		},
	})

	events = append(events, AnthropicStreamEvent{
		Type:  "content_block_delta",
		Index: &idx,
		Delta: &AnthropicDelta{
			Type:      "signature_delta",
			Signature: encodeCompactionItemSignature(*evt.Item),
		},
	})

	return events
}

// resToAnthHandleWebSearchDone converts an OpenAI web_search_call output item
// into Anthropic server_tool_use + web_search_tool_result content block pairs.
// This allows Claude Code to count the searches performed.
func resToAnthHandleWebSearchDone(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	var events []AnthropicStreamEvent
	events = append(events, closeCurrentBlock(state)...)

	toolUseID := "srvtoolu_" + evt.Item.ID
	query := ""
	if evt.Item.Action != nil {
		query = evt.Item.Action.Query
	}
	inputJSON, _ := json.Marshal(map[string]string{"query": query})

	// Emit server_tool_use block (start + stop).
	idx1 := state.ContentBlockIndex
	events = append(events, AnthropicStreamEvent{
		Type:  "content_block_start",
		Index: &idx1,
		ContentBlock: &AnthropicContentBlock{
			Type:  "server_tool_use",
			ID:    toolUseID,
			Name:  "web_search",
			Input: inputJSON,
		},
	})
	events = append(events, AnthropicStreamEvent{
		Type:  "content_block_stop",
		Index: &idx1,
	})
	state.ContentBlockIndex++

	// Emit web_search_tool_result block (start + stop).
	// Content is empty because OpenAI does not expose individual search results;
	// the model consumes them internally and produces text output.
	emptyResults, _ := json.Marshal([]struct{}{})
	idx2 := state.ContentBlockIndex
	events = append(events, AnthropicStreamEvent{
		Type:  "content_block_start",
		Index: &idx2,
		ContentBlock: &AnthropicContentBlock{
			Type:      "web_search_tool_result",
			ToolUseID: toolUseID,
			Content:   emptyResults,
		},
	})
	events = append(events, AnthropicStreamEvent{
		Type:  "content_block_stop",
		Index: &idx2,
	})
	state.ContentBlockIndex++

	return events
}

func resToAnthHandleCompleted(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	if state.MessageStopSent {
		return nil
	}

	var events []AnthropicStreamEvent
	events = append(events, closeCurrentBlock(state)...)

	stopReason := "end_turn"
	if evt.Response != nil {
		if evt.Response.Usage != nil {
			usage := anthropicUsageFromResponsesUsage(evt.Response.Usage)
			state.InputTokens = usage.InputTokens
			state.OutputTokens = usage.OutputTokens
			state.CacheReadInputTokens = usage.CacheReadInputTokens
		}
		switch evt.Response.Status {
		case "incomplete":
			if evt.Response.IncompleteDetails != nil && evt.Response.IncompleteDetails.Reason == "max_output_tokens" {
				stopReason = "max_tokens"
			}
		case "completed":
			if state.ContentBlockIndex > 0 && state.CurrentBlockType == "tool_use" {
				stopReason = "tool_use"
			}
		}
	}

	events = append(events,
		AnthropicStreamEvent{
			Type: "message_delta",
			Delta: &AnthropicDelta{
				StopReason: stopReason,
			},
			Usage: &AnthropicUsage{
				InputTokens:          state.InputTokens,
				OutputTokens:         state.OutputTokens,
				CacheReadInputTokens: state.CacheReadInputTokens,
			},
		},
		AnthropicStreamEvent{Type: "message_stop"},
	)
	state.MessageStopSent = true
	return events
}

func closeCurrentBlock(state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	if !state.ContentBlockOpen {
		return nil
	}
	idx := state.ContentBlockIndex
	state.ContentBlockOpen = false
	state.ContentBlockIndex++
	state.CurrentToolName = ""
	state.CurrentToolArgs = ""
	return []AnthropicStreamEvent{{
		Type:  "content_block_stop",
		Index: &idx,
	}}
}

func resToAnthHandleFailed(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	if state.MessageStopSent {
		return nil
	}

	var events []AnthropicStreamEvent
	events = append(events, closeCurrentBlock(state)...)

	msg := "The response failed due to an unknown error."
	if evt.Response != nil && evt.Response.Error != nil {
		msg = evt.Response.Error.Message
	}

	events = append(events, AnthropicStreamEvent{
		Type: "error",
		Error: &AnthropicErrorDetail{
			Type:    "api_error",
			Message: msg,
		},
	})
	state.MessageStopSent = true
	return events
}

func resToAnthHandleError(evt *ResponsesStreamEvent, state *ResponsesEventToAnthropicState) []AnthropicStreamEvent {
	msg := "An unexpected error occurred during streaming."
	if evt.Delta != "" {
		msg = evt.Delta
	}

	state.MessageStopSent = true
	return []AnthropicStreamEvent{{
		Type: "error",
		Error: &AnthropicErrorDetail{
			Type:    "api_error",
			Message: msg,
		},
	}}
}
