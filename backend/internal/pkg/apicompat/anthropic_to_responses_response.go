package apicompat

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"
)

// ---------------------------------------------------------------------------
// Non-streaming: AnthropicResponse → ResponsesResponse
// ---------------------------------------------------------------------------

// AnthropicToResponsesResponse converts an Anthropic Messages response into a
// Responses API response. This is the reverse of ResponsesToAnthropic and
// enables Anthropic upstream responses to be returned in OpenAI Responses format.
func AnthropicToResponsesResponse(resp *AnthropicResponse) *ResponsesResponse {
	id := resp.ID
	if id == "" {
		id = generateResponsesID()
	}

	out := &ResponsesResponse{
		ID:     id,
		Object: "response",
		Model:  resp.Model,
	}
	if len(resp.Container) > 0 {
		if conversation := anthropicContainerToResponsesConversation(resp.Container); conversation != nil {
			out.Conversation = conversation
		}
	}

	var outputs []ResponsesOutput
	var msgParts []ResponsesContentPart

	for _, block := range resp.Content {
		switch block.Type {
		case "thinking":
			if block.Signature != "" {
				// Check for compaction first
				if compaction := decodeCompactionSignature(block.Signature); compaction != nil {
					outputs = append(outputs, ResponsesOutput{
						Type:             "compaction",
						ID:               compaction.id,
						EncryptedContent: compaction.encryptedContent,
					})
					continue
				}
				// Reasoning with signature
				enc, id := parseReasoningSignature(block.Signature)
				if id != "" && len(id) <= maxReasoningIDLength {
					thinking := block.Thinking
					if thinking == thinkingPlaceholder {
						thinking = ""
					}
					item := ResponsesOutput{
						Type:             "reasoning",
						ID:               id,
						EncryptedContent: enc,
					}
					if thinking != "" {
						item.Summary = []ResponsesSummary{{Type: "summary_text", Text: thinking}}
					} else {
						item.Summary = []ResponsesSummary{{Type: "summary_text", Text: ""}}
					}
					outputs = append(outputs, item)
					continue
				}
			}
			// Fallback: thinking without valid signature
			if block.Thinking != "" {
				outputs = append(outputs, ResponsesOutput{
					Type: "reasoning",
					ID:   generateItemID(),
					Summary: []ResponsesSummary{{
						Type: "summary_text",
						Text: block.Thinking,
					}},
				})
			}
		case "text":
			if block.Text != "" {
				msgParts = append(msgParts, ResponsesContentPart{
					Type: "output_text",
					Text: block.Text,
				})
			}
		case "tool_use", "mcp_tool_use":
			args := "{}"
			if len(block.Input) > 0 {
				args = string(block.Input)
			}
			item := ResponsesOutput{
				Type:      "function_call",
				ID:        generateItemID(),
				CallID:    block.ID,
				Name:      block.Name,
				Arguments: args,
				Status:    "completed",
			}
			if block.Type == "mcp_tool_use" {
				item.Namespace = block.ServerName
				if item.Namespace == "" {
					item.Namespace = "mcp"
				}
				item.RawItem = mustMarshalAnthropicContentBlock(block)
			}
			outputs = append(outputs, item)
		}
	}

	// Assemble message output item from text parts
	if len(msgParts) > 0 {
		outputs = append(outputs, ResponsesOutput{
			Type:    "message",
			ID:      generateItemID(),
			Role:    "assistant",
			Content: msgParts,
			Status:  "completed",
		})
	}

	if len(outputs) == 0 {
		outputs = append(outputs, ResponsesOutput{
			Type:    "message",
			ID:      generateItemID(),
			Role:    "assistant",
			Content: []ResponsesContentPart{{Type: "output_text", Text: ""}},
			Status:  "completed",
		})
	}
	out.Output = outputs

	// Map stop_reason → status
	out.Status = anthropicStopReasonToResponsesStatus(resp.StopReason, resp.Content)
	if out.Status == "incomplete" {
		out.IncompleteDetails = &ResponsesIncompleteDetails{Reason: "max_output_tokens"}
	}

	// Usage
	out.Usage = &ResponsesUsage{
		InputTokens:  resp.Usage.InputTokens,
		OutputTokens: resp.Usage.OutputTokens,
		TotalTokens:  resp.Usage.InputTokens + resp.Usage.OutputTokens,
	}
	if resp.Usage.CacheReadInputTokens > 0 {
		out.Usage.InputTokensDetails = &ResponsesInputTokensDetails{
			CachedTokens: resp.Usage.CacheReadInputTokens,
		}
	}

	return out
}

func anthropicContainerToResponsesConversation(raw json.RawMessage) *ResponsesConversation {
	if len(raw) == 0 {
		return nil
	}
	var conversation ResponsesConversation
	if err := json.Unmarshal(raw, &conversation); err == nil && conversation.ID != "" {
		return &conversation
	}
	var id string
	if err := json.Unmarshal(raw, &id); err == nil && id != "" {
		return &ResponsesConversation{ID: id}
	}
	return nil
}

// anthropicStopReasonToResponsesStatus maps Anthropic stop_reason to Responses status.
func anthropicStopReasonToResponsesStatus(stopReason string, blocks []AnthropicContentBlock) string {
	switch stopReason {
	case "max_tokens":
		return "incomplete"
	case "end_turn", "tool_use", "stop_sequence":
		return "completed"
	default:
		return "completed"
	}
}

// ---------------------------------------------------------------------------
// Streaming: AnthropicStreamEvent → []ResponsesStreamEvent (stateful converter)
// ---------------------------------------------------------------------------

// AnthropicEventToResponsesState tracks state for converting a sequence of
// Anthropic SSE events into Responses SSE events.
type AnthropicEventToResponsesState struct {
	ResponseID     string
	Model          string
	Created        int64
	SequenceNumber int

	// CreatedSent tracks whether response.created has been emitted.
	CreatedSent bool
	// CompletedSent tracks whether the terminal event has been emitted.
	CompletedSent bool

	// Current output tracking
	OutputIndex     int
	CurrentItemID   string
	CurrentItemType string // "message" | "function_call" | "reasoning"
	CurrentItem     *ResponsesOutput

	// For message output: accumulate text parts
	ContentIndex int

	// For function_call: track per-output info
	CurrentCallID string
	CurrentName   string

	// For reasoning: accumulate signature deltas
	AccumulatedSignature string

	// Final response accumulation
	Output     []ResponsesOutput
	StopReason string

	// Usage from message_delta
	InputTokens          int
	OutputTokens         int
	CacheReadInputTokens int
}

// NewAnthropicEventToResponsesState returns an initialised stream state.
func NewAnthropicEventToResponsesState() *AnthropicEventToResponsesState {
	return &AnthropicEventToResponsesState{
		Created: time.Now().Unix(),
	}
}

// AnthropicEventToResponsesEvents converts a single Anthropic SSE event into
// zero or more Responses SSE events, updating state as it goes.
func AnthropicEventToResponsesEvents(
	evt *AnthropicStreamEvent,
	state *AnthropicEventToResponsesState,
) []ResponsesStreamEvent {
	switch evt.Type {
	case "message_start":
		return anthToResHandleMessageStart(evt, state)
	case "content_block_start":
		return anthToResHandleContentBlockStart(evt, state)
	case "content_block_delta":
		return anthToResHandleContentBlockDelta(evt, state)
	case "content_block_stop":
		return anthToResHandleContentBlockStop(evt, state)
	case "message_delta":
		return anthToResHandleMessageDelta(evt, state)
	case "message_stop":
		return anthToResHandleMessageStop(state)
	default:
		return nil
	}
}

// FinalizeAnthropicResponsesStream emits synthetic termination events if the
// stream ended without a proper message_stop.
func FinalizeAnthropicResponsesStream(state *AnthropicEventToResponsesState) []ResponsesStreamEvent {
	if !state.CreatedSent || state.CompletedSent {
		return nil
	}

	var events []ResponsesStreamEvent
	events = append(events, closeCurrentResponsesItem(state)...)

	status := anthropicStopReasonToResponsesStatus(state.StopReason, nil)
	var incompleteDetails *ResponsesIncompleteDetails
	if status == "incomplete" {
		incompleteDetails = &ResponsesIncompleteDetails{Reason: "max_output_tokens"}
	}

	events = append(events, makeResponsesCompletedEvent(state, status, incompleteDetails))
	state.CompletedSent = true
	return events
}

// ResponsesEventToSSE formats a ResponsesStreamEvent as an SSE data line.
func ResponsesEventToSSE(evt ResponsesStreamEvent) (string, error) {
	data, err := json.Marshal(evt)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("event: %s\ndata: %s\n\n", evt.Type, data), nil
}

// --- internal handlers ---

func anthToResHandleMessageStart(evt *AnthropicStreamEvent, state *AnthropicEventToResponsesState) []ResponsesStreamEvent {
	if evt.Message != nil {
		state.ResponseID = evt.Message.ID
		if state.Model == "" {
			state.Model = evt.Message.Model
		}
		if evt.Message.Usage.InputTokens > 0 {
			state.InputTokens = evt.Message.Usage.InputTokens
		}
	}

	if state.CreatedSent {
		return nil
	}
	state.CreatedSent = true

	return []ResponsesStreamEvent{makeResponsesCreatedEvent(state)}
}

func anthToResHandleContentBlockStart(evt *AnthropicStreamEvent, state *AnthropicEventToResponsesState) []ResponsesStreamEvent {
	if evt.ContentBlock == nil {
		return nil
	}

	var events []ResponsesStreamEvent

	switch evt.ContentBlock.Type {
	case "thinking":
		events = append(events, closeCurrentResponsesItem(state)...)

		state.CurrentItemID = generateItemID()
		state.CurrentItemType = "reasoning"
		state.ContentIndex = 0
		state.CurrentCallID = ""
		state.CurrentName = ""
		state.CurrentItem = &ResponsesOutput{
			Type:   "reasoning",
			ID:     state.CurrentItemID,
			Status: "in_progress",
		}

		events = append(events, makeResponsesEvent(state, "response.output_item.added", &ResponsesStreamEvent{
			OutputIndex: state.OutputIndex,
			Item:        cloneResponsesOutput(state.CurrentItem),
		}))

	case "text":
		if state.CurrentItemType != "message" {
			events = append(events, closeCurrentResponsesItem(state)...)

			state.CurrentItemID = generateItemID()
			state.CurrentItemType = "message"
			state.ContentIndex = 0
			state.CurrentCallID = ""
			state.CurrentName = ""
			state.CurrentItem = &ResponsesOutput{
				Type:   "message",
				ID:     state.CurrentItemID,
				Role:   "assistant",
				Status: "in_progress",
			}

			events = append(events, makeResponsesEvent(state, "response.output_item.added", &ResponsesStreamEvent{
				OutputIndex: state.OutputIndex,
				Item:        cloneResponsesOutput(state.CurrentItem),
			}))
		}

		if state.CurrentItem == nil {
			state.CurrentItem = &ResponsesOutput{
				Type:   "message",
				ID:     state.CurrentItemID,
				Role:   "assistant",
				Status: "in_progress",
			}
		}
		state.CurrentItem.Content = append(state.CurrentItem.Content, ResponsesContentPart{Type: "output_text"})
		state.ContentIndex = len(state.CurrentItem.Content) - 1

		events = append(events, makeResponsesEvent(state, "response.content_part.added", &ResponsesStreamEvent{
			OutputIndex:  state.OutputIndex,
			ContentIndex: state.ContentIndex,
			ItemID:       state.CurrentItemID,
			Part:         mustMarshalResponsesPart(ResponsesOutputPart{Type: "output_text"}),
		}))

	case "tool_use", "mcp_tool_use":
		events = append(events, closeCurrentResponsesItem(state)...)

		state.CurrentItemID = generateItemID()
		state.CurrentItemType = "function_call"
		state.CurrentCallID = evt.ContentBlock.ID
		state.CurrentName = evt.ContentBlock.Name
		state.ContentIndex = 0
		state.CurrentItem = &ResponsesOutput{
			Type:   "function_call",
			ID:     state.CurrentItemID,
			CallID: state.CurrentCallID,
			Name:   state.CurrentName,
			Status: "in_progress",
		}
		if evt.ContentBlock.Type == "mcp_tool_use" {
			state.CurrentItem.Namespace = evt.ContentBlock.ServerName
			if state.CurrentItem.Namespace == "" {
				state.CurrentItem.Namespace = "mcp"
			}
			state.CurrentItem.RawItem = mustMarshalAnthropicContentBlock(*evt.ContentBlock)
		}

		events = append(events, makeResponsesEvent(state, "response.output_item.added", &ResponsesStreamEvent{
			OutputIndex: state.OutputIndex,
			Item:        cloneResponsesOutput(state.CurrentItem),
		}))
	}

	return events
}

func anthToResHandleContentBlockDelta(evt *AnthropicStreamEvent, state *AnthropicEventToResponsesState) []ResponsesStreamEvent {
	if evt.Delta == nil {
		return nil
	}

	switch evt.Delta.Type {
	case "text_delta":
		if evt.Delta.Text == "" {
			return nil
		}
		if state.CurrentItem != nil && state.CurrentItemType == "message" && state.ContentIndex < len(state.CurrentItem.Content) {
			state.CurrentItem.Content[state.ContentIndex].Text += evt.Delta.Text
		}
		return []ResponsesStreamEvent{makeResponsesEvent(state, "response.output_text.delta", &ResponsesStreamEvent{
			OutputIndex:  state.OutputIndex,
			ContentIndex: state.ContentIndex,
			Delta:        evt.Delta.Text,
			ItemID:       state.CurrentItemID,
		})}

	case "thinking_delta":
		if evt.Delta.Thinking == "" {
			return nil
		}
		if state.CurrentItem != nil && state.CurrentItemType == "reasoning" {
			if len(state.CurrentItem.Summary) == 0 {
				state.CurrentItem.Summary = []ResponsesSummary{{Type: "summary_text"}}
			}
			state.CurrentItem.Summary[0].Text += evt.Delta.Thinking
		}
		return []ResponsesStreamEvent{makeResponsesEvent(state, "response.reasoning_text.delta", &ResponsesStreamEvent{
			OutputIndex:  state.OutputIndex,
			SummaryIndex: 0,
			Delta:        evt.Delta.Thinking,
			ItemID:       state.CurrentItemID,
		})}

	case "input_json_delta":
		if evt.Delta.PartialJSON == "" {
			return nil
		}
		if state.CurrentItem != nil && state.CurrentItemType == "function_call" {
			state.CurrentItem.Arguments += evt.Delta.PartialJSON
		}
		return []ResponsesStreamEvent{makeResponsesEvent(state, "response.function_call_arguments.delta", &ResponsesStreamEvent{
			OutputIndex: state.OutputIndex,
			Delta:       evt.Delta.PartialJSON,
			ItemID:      state.CurrentItemID,
			CallID:      state.CurrentCallID,
			Name:        state.CurrentName,
		})}

	case "signature_delta":
		state.AccumulatedSignature += evt.Delta.Signature
		return nil
	}

	return nil
}

func anthToResHandleContentBlockStop(_ *AnthropicStreamEvent, state *AnthropicEventToResponsesState) []ResponsesStreamEvent {
	switch state.CurrentItemType {
	case "reasoning":
		text := ""
		if state.CurrentItem != nil && len(state.CurrentItem.Summary) > 0 {
			text = state.CurrentItem.Summary[0].Text
		}
		events := []ResponsesStreamEvent{
			makeResponsesEvent(state, "response.reasoning_text.done", &ResponsesStreamEvent{
				OutputIndex:  state.OutputIndex,
				SummaryIndex: 0,
				ItemID:       state.CurrentItemID,
				Text:         text,
			}),
		}
		events = append(events, closeCurrentResponsesItem(state)...)
		return events

	case "function_call":
		arguments := ""
		if state.CurrentItem != nil {
			arguments = state.CurrentItem.Arguments
		}
		events := []ResponsesStreamEvent{
			makeResponsesEvent(state, "response.function_call_arguments.done", &ResponsesStreamEvent{
				OutputIndex: state.OutputIndex,
				ItemID:      state.CurrentItemID,
				CallID:      state.CurrentCallID,
				Name:        state.CurrentName,
				Arguments:   arguments,
			}),
		}
		events = append(events, closeCurrentResponsesItem(state)...)
		return events

	case "message":
		part := ResponsesContentPart{Type: "output_text"}
		if state.CurrentItem != nil && state.ContentIndex < len(state.CurrentItem.Content) {
			part = state.CurrentItem.Content[state.ContentIndex]
		}
		return []ResponsesStreamEvent{
			makeResponsesEvent(state, "response.output_text.done", &ResponsesStreamEvent{
				OutputIndex:  state.OutputIndex,
				ContentIndex: state.ContentIndex,
				ItemID:       state.CurrentItemID,
				Text:         part.Text,
			}),
			makeResponsesEvent(state, "response.content_part.done", &ResponsesStreamEvent{
				OutputIndex:  state.OutputIndex,
				ContentIndex: state.ContentIndex,
				ItemID:       state.CurrentItemID,
				Part:         mustMarshalResponsesPart(part),
			}),
		}
	}

	return nil
}

func anthToResHandleMessageDelta(evt *AnthropicStreamEvent, state *AnthropicEventToResponsesState) []ResponsesStreamEvent {
	if evt.Usage != nil {
		state.OutputTokens = evt.Usage.OutputTokens
		if evt.Usage.CacheReadInputTokens > 0 {
			state.CacheReadInputTokens = evt.Usage.CacheReadInputTokens
		}
	}
	if evt.Delta != nil && evt.Delta.StopReason != "" {
		state.StopReason = evt.Delta.StopReason
	}

	return nil
}

func anthToResHandleMessageStop(state *AnthropicEventToResponsesState) []ResponsesStreamEvent {
	if state.CompletedSent {
		return nil
	}

	var events []ResponsesStreamEvent
	events = append(events, closeCurrentResponsesItem(state)...)

	status := anthropicStopReasonToResponsesStatus(state.StopReason, nil)
	var incompleteDetails *ResponsesIncompleteDetails
	if status == "incomplete" {
		incompleteDetails = &ResponsesIncompleteDetails{Reason: "max_output_tokens"}
	}

	events = append(events, makeResponsesCompletedEvent(state, status, incompleteDetails))
	state.CompletedSent = true
	return events
}

func closeCurrentResponsesItem(state *AnthropicEventToResponsesState) []ResponsesStreamEvent {
	if state.CurrentItemType == "" || state.CurrentItem == nil {
		state.CurrentItemType = ""
		state.CurrentItemID = ""
		state.CurrentCallID = ""
		state.CurrentName = ""
		state.AccumulatedSignature = ""
		return nil
	}

	item := cloneResponsesOutput(state.CurrentItem)
	item.Status = "completed"

	if item.Type == "reasoning" && state.AccumulatedSignature != "" {
		sig := state.AccumulatedSignature
		if compaction := decodeCompactionSignature(sig); compaction != nil {
			item.Type = "compaction"
			item.ID = compaction.id
			item.EncryptedContent = compaction.encryptedContent
			item.Summary = nil
		} else {
			enc, id := parseReasoningSignature(sig)
			item.EncryptedContent = enc
			if id != "" {
				item.ID = id
			}
		}
	}

	state.Output = append(state.Output, *item)
	completedIndex := state.OutputIndex

	state.CurrentItem = nil
	state.CurrentItemType = ""
	state.CurrentItemID = ""
	state.CurrentCallID = ""
	state.CurrentName = ""
	state.AccumulatedSignature = ""
	state.OutputIndex++
	state.ContentIndex = 0

	return []ResponsesStreamEvent{makeResponsesEvent(state, "response.output_item.done", &ResponsesStreamEvent{
		OutputIndex: completedIndex,
		Item:        item,
	})}
}

func makeResponsesCreatedEvent(state *AnthropicEventToResponsesState) ResponsesStreamEvent {
	seq := state.SequenceNumber
	state.SequenceNumber++
	return ResponsesStreamEvent{
		Type:           "response.created",
		SequenceNumber: seq,
		Response: &ResponsesResponse{
			ID:        state.ResponseID,
			Object:    "response",
			CreatedAt: state.Created,
			Model:     state.Model,
			Status:    "in_progress",
			Output:    []ResponsesOutput{},
		},
	}
}

func makeResponsesCompletedEvent(
	state *AnthropicEventToResponsesState,
	status string,
	incompleteDetails *ResponsesIncompleteDetails,
) ResponsesStreamEvent {
	seq := state.SequenceNumber
	state.SequenceNumber++

	usage := &ResponsesUsage{
		InputTokens:  state.InputTokens,
		OutputTokens: state.OutputTokens,
		TotalTokens:  state.InputTokens + state.OutputTokens,
	}
	if state.CacheReadInputTokens > 0 {
		usage.InputTokensDetails = &ResponsesInputTokensDetails{
			CachedTokens: state.CacheReadInputTokens,
		}
	}

	output := append([]ResponsesOutput(nil), state.Output...)
	eventType := "response.completed"
	if status == "incomplete" {
		eventType = "response.incomplete"
	}

	return ResponsesStreamEvent{
		Type:           eventType,
		SequenceNumber: seq,
		Response: &ResponsesResponse{
			ID:                state.ResponseID,
			Object:            "response",
			CreatedAt:         state.Created,
			Model:             state.Model,
			Status:            status,
			Output:            output,
			OutputText:        collectResponsesOutputText(output),
			Usage:             usage,
			IncompleteDetails: incompleteDetails,
		},
	}
}

func makeResponsesEvent(state *AnthropicEventToResponsesState, eventType string, template *ResponsesStreamEvent) ResponsesStreamEvent {
	seq := state.SequenceNumber
	state.SequenceNumber++

	evt := *template
	evt.Type = eventType
	evt.SequenceNumber = seq
	return evt
}

func cloneResponsesOutput(item *ResponsesOutput) *ResponsesOutput {
	if item == nil {
		return nil
	}
	cloned := *item
	if item.Content != nil {
		cloned.Content = append([]ResponsesContentPart(nil), item.Content...)
	}
	if item.Summary != nil {
		cloned.Summary = append([]ResponsesSummary(nil), item.Summary...)
	}
	return &cloned
}

func collectResponsesOutputText(output []ResponsesOutput) string {
	var text string
	for _, item := range output {
		if item.Type != "message" {
			continue
		}
		for _, part := range item.Content {
			switch part.Type {
			case "output_text":
				text += part.Text
			case "refusal":
				text += part.Refusal
			}
		}
	}
	return text
}

func mustMarshalResponsesPart(part any) json.RawMessage {
	payload, err := json.Marshal(part)
	if err != nil {
		return nil
	}
	return payload
}

func generateResponsesID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "resp_" + hex.EncodeToString(b)
}

func generateItemID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "item_" + hex.EncodeToString(b)
}
