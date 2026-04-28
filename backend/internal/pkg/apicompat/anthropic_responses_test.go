package apicompat

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func stripEmptyAnthropicFields(raw json.RawMessage) string {
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return string(raw)
	}
	for k, v := range m {
		if s, ok := v.(string); ok && s == "" {
			delete(m, k)
		}
	}
	out, _ := json.Marshal(m)
	return string(out)
}

// ---------------------------------------------------------------------------
// AnthropicToResponses tests
// ---------------------------------------------------------------------------

func TestAnthropicToResponses_BasicText(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Stream:    true,
		Messages: []AnthropicMessage{
			{Role: "user", Content: json.RawMessage(`"Hello"`)},
		},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	assert.Equal(t, "gpt-5.2", resp.Model)
	assert.True(t, resp.Stream)
	assert.Equal(t, 12800, *resp.MaxOutputTokens)
	assert.False(t, *resp.Store)

	var items []ResponsesInputItem
	require.NoError(t, json.Unmarshal(resp.Input, &items))
	require.Len(t, items, 1)
	assert.Equal(t, "user", items[0].Role)
}

func TestAnthropicToResponses_SystemPrompt(t *testing.T) {
	t.Run("string", func(t *testing.T) {
		req := &AnthropicRequest{
			Model:     "gpt-5.2",
			MaxTokens: 100,
			System:    json.RawMessage(`"You are helpful."`),
			Messages:  []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hi"`)}},
		}
		resp, err := AnthropicToResponses(req)
		require.NoError(t, err)

		assert.Equal(t, "You are helpful.", resp.Instructions)

		var items []ResponsesInputItem
		require.NoError(t, json.Unmarshal(resp.Input, &items))
		require.Len(t, items, 1)
		assert.Equal(t, "user", items[0].Role)
	})

	t.Run("array", func(t *testing.T) {
		req := &AnthropicRequest{
			Model:     "gpt-5.2",
			MaxTokens: 100,
			System:    json.RawMessage(`[{"type":"text","text":"Part 1"},{"type":"text","text":"Part 2"}]`),
			Messages:  []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hi"`)}},
		}
		resp, err := AnthropicToResponses(req)
		require.NoError(t, err)

		assert.Equal(t, "Part 1\n\nPart 2", resp.Instructions)

		var items []ResponsesInputItem
		require.NoError(t, json.Unmarshal(resp.Input, &items))
		require.Len(t, items, 1)
		assert.Equal(t, "user", items[0].Role)
	})
}

func TestAnthropicToResponses_ToolUse(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages: []AnthropicMessage{
			{Role: "user", Content: json.RawMessage(`"What is the weather?"`)},
			{Role: "assistant", Content: json.RawMessage(`[{"type":"text","text":"Let me check."},{"type":"tool_use","id":"call_1","name":"get_weather","input":{"city":"NYC"}}]`)},
			{Role: "user", Content: json.RawMessage(`[{"type":"tool_result","tool_use_id":"call_1","content":"Sunny, 72°F"}]`)},
		},
		Tools: []AnthropicTool{
			{Name: "get_weather", Description: "Get weather", InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)},
		},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)

	// Check tools
	require.Len(t, resp.Tools, 1)
	assert.Equal(t, "function", resp.Tools[0].Type)
	assert.Equal(t, "get_weather", resp.Tools[0].Name)

	// Check input items
	var items []ResponsesInputItem
	require.NoError(t, json.Unmarshal(resp.Input, &items))
	// user + assistant + function_call + function_call_output = 4
	require.Len(t, items, 4)

	assert.Equal(t, "user", items[0].Role)
	assert.Equal(t, "assistant", items[1].Role)
	assert.Equal(t, "function_call", items[2].Type)
	assert.Equal(t, "call_1", items[2].CallID)
	assert.Empty(t, items[2].ID)
	assert.Equal(t, "function_call_output", items[3].Type)
	assert.Equal(t, "call_1", items[3].CallID)
	assert.Equal(t, "Sunny, 72°F", items[3].Output)
}

func TestAnthropicToResponses_ThinkingIgnored(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages: []AnthropicMessage{
			{Role: "user", Content: json.RawMessage(`"Hello"`)},
			{Role: "assistant", Content: json.RawMessage(`[{"type":"thinking","thinking":"deep thought"},{"type":"text","text":"Hi!"}]`)},
			{Role: "user", Content: json.RawMessage(`"More"`)},
		},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)

	var items []ResponsesInputItem
	require.NoError(t, json.Unmarshal(resp.Input, &items))
	// user + assistant(text only, thinking ignored) + user = 3
	require.Len(t, items, 3)
	assert.Equal(t, "assistant", items[1].Role)
	// Assistant content should only have text, not thinking.
	var parts []ResponsesContentPart
	require.NoError(t, json.Unmarshal(items[1].Content, &parts))
	require.Len(t, parts, 1)
	assert.Equal(t, "output_text", parts[0].Type)
	assert.Equal(t, "Hi!", parts[0].Text)
}

func TestAnthropicToResponses_MaxTokensFloor(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 10, // below minAnthropicMaxOutputTokens (12800)
		Messages:  []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hi"`)}},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	assert.Equal(t, 12800, *resp.MaxOutputTokens)
}

// ---------------------------------------------------------------------------
// ResponsesToAnthropic (non-streaming) tests
// ---------------------------------------------------------------------------

func TestResponsesToAnthropic_TextOnly(t *testing.T) {
	resp := &ResponsesResponse{
		ID:     "resp_123",
		Model:  "gpt-5.2",
		Status: "completed",
		Output: []ResponsesOutput{
			{
				Type: "message",
				Content: []ResponsesContentPart{
					{Type: "output_text", Text: "Hello there!"},
				},
			},
		},
		Usage: &ResponsesUsage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
	}

	anth := ResponsesToAnthropic(resp, "claude-opus-4-6")
	assert.Equal(t, "resp_123", anth.ID)
	assert.Equal(t, "claude-opus-4-6", anth.Model)
	assert.Equal(t, "end_turn", anth.StopReason)
	require.Len(t, anth.Content, 1)
	assert.Equal(t, "text", anth.Content[0].Type)
	assert.Equal(t, "Hello there!", anth.Content[0].Text)
	assert.Equal(t, 10, anth.Usage.InputTokens)
	assert.Equal(t, 5, anth.Usage.OutputTokens)
}

func TestResponsesToAnthropic_CachedTokensUseAnthropicInputSemantics(t *testing.T) {
	resp := &ResponsesResponse{
		ID:     "resp_cached",
		Model:  "gpt-5.2",
		Status: "completed",
		Output: []ResponsesOutput{
			{
				Type: "message",
				Content: []ResponsesContentPart{
					{Type: "output_text", Text: "Cached response"},
				},
			},
		},
		Usage: &ResponsesUsage{
			InputTokens:  54006,
			OutputTokens: 123,
			TotalTokens:  54129,
			InputTokensDetails: &ResponsesInputTokensDetails{
				CachedTokens: 50688,
			},
		},
	}

	anth := ResponsesToAnthropic(resp, "claude-sonnet-4-5-20250929")
	assert.Equal(t, 3318, anth.Usage.InputTokens)
	assert.Equal(t, 50688, anth.Usage.CacheReadInputTokens)
	assert.Equal(t, 123, anth.Usage.OutputTokens)
}

func TestResponsesToAnthropic_CachedTokensClampInputTokens(t *testing.T) {
	resp := &ResponsesResponse{
		ID:     "resp_cached_clamp",
		Model:  "gpt-5.2",
		Status: "completed",
		Usage: &ResponsesUsage{
			InputTokens:  100,
			OutputTokens: 5,
			InputTokensDetails: &ResponsesInputTokensDetails{
				CachedTokens: 150,
			},
		},
	}

	anth := ResponsesToAnthropic(resp, "claude-sonnet-4-5-20250929")
	assert.Equal(t, 0, anth.Usage.InputTokens)
	assert.Equal(t, 150, anth.Usage.CacheReadInputTokens)
	assert.Equal(t, 5, anth.Usage.OutputTokens)
}

func TestResponsesToAnthropic_ToolUse(t *testing.T) {
	resp := &ResponsesResponse{
		ID:     "resp_456",
		Model:  "gpt-5.2",
		Status: "completed",
		Output: []ResponsesOutput{
			{
				Type: "message",
				Content: []ResponsesContentPart{
					{Type: "output_text", Text: "Let me check."},
				},
			},
			{
				Type:      "function_call",
				CallID:    "call_1",
				Name:      "get_weather",
				Arguments: `{"city":"NYC"}`,
			},
		},
	}

	anth := ResponsesToAnthropic(resp, "claude-opus-4-6")
	assert.Equal(t, "tool_use", anth.StopReason)
	require.Len(t, anth.Content, 2)
	assert.Equal(t, "text", anth.Content[0].Type)
	assert.Equal(t, "tool_use", anth.Content[1].Type)
	assert.Equal(t, "call_1", anth.Content[1].ID)
	assert.Equal(t, "get_weather", anth.Content[1].Name)
	assert.JSONEq(t, `{"city":"NYC"}`, string(anth.Content[1].Input))
}

func TestResponsesToAnthropic_ReadToolDropsEmptyPages(t *testing.T) {
	resp := &ResponsesResponse{
		ID:     "resp_read",
		Model:  "gpt-5.5",
		Status: "completed",
		Output: []ResponsesOutput{
			{
				Type:      "function_call",
				CallID:    "call_read",
				Name:      "Read",
				Arguments: `{"file_path":"/tmp/demo.py","limit":2000,"offset":0,"pages":""}`,
			},
		},
	}

	anth := ResponsesToAnthropic(resp, "claude-opus-4-6")
	require.Len(t, anth.Content, 1)
	assert.Equal(t, "tool_use", anth.Content[0].Type)
	assert.JSONEq(t, `{"file_path":"/tmp/demo.py","limit":2000,"offset":0}`, string(anth.Content[0].Input))
}

func TestResponsesToAnthropic_PreservesEmptyStringsForOtherTools(t *testing.T) {
	resp := &ResponsesResponse{
		ID:     "resp_other",
		Model:  "gpt-5.5",
		Status: "completed",
		Output: []ResponsesOutput{
			{
				Type:      "function_call",
				CallID:    "call_other",
				Name:      "Search",
				Arguments: `{"query":""}`,
			},
		},
	}

	anth := ResponsesToAnthropic(resp, "claude-opus-4-6")
	require.Len(t, anth.Content, 1)
	assert.JSONEq(t, `{"query":""}`, string(anth.Content[0].Input))
}

func TestResponsesToAnthropic_Reasoning(t *testing.T) {
	resp := &ResponsesResponse{
		ID:     "resp_789",
		Model:  "gpt-5.2",
		Status: "completed",
		Output: []ResponsesOutput{
			{
				Type: "reasoning",
				Summary: []ResponsesSummary{
					{Type: "summary_text", Text: "Thinking about the answer..."},
				},
			},
			{
				Type: "message",
				Content: []ResponsesContentPart{
					{Type: "output_text", Text: "42"},
				},
			},
		},
	}

	anth := ResponsesToAnthropic(resp, "claude-opus-4-6")
	require.Len(t, anth.Content, 2)
	assert.Equal(t, "thinking", anth.Content[0].Type)
	assert.Equal(t, "Thinking about the answer...", anth.Content[0].Thinking)
	assert.Equal(t, "text", anth.Content[1].Type)
	assert.Equal(t, "42", anth.Content[1].Text)
}

func TestResponsesToAnthropic_Incomplete(t *testing.T) {
	resp := &ResponsesResponse{
		ID:     "resp_inc",
		Model:  "gpt-5.2",
		Status: "incomplete",
		IncompleteDetails: &ResponsesIncompleteDetails{
			Reason: "max_output_tokens",
		},
		Output: []ResponsesOutput{
			{
				Type:    "message",
				Content: []ResponsesContentPart{{Type: "output_text", Text: "Partial..."}},
			},
		},
	}

	anth := ResponsesToAnthropic(resp, "claude-opus-4-6")
	assert.Equal(t, "max_tokens", anth.StopReason)
}

func TestResponsesToAnthropic_EmptyOutput(t *testing.T) {
	resp := &ResponsesResponse{
		ID:     "resp_empty",
		Model:  "gpt-5.2",
		Status: "completed",
		Output: []ResponsesOutput{},
	}

	anth := ResponsesToAnthropic(resp, "claude-opus-4-6")
	require.Len(t, anth.Content, 1)
	assert.Equal(t, "text", anth.Content[0].Type)
	assert.Equal(t, "", anth.Content[0].Text)
}

// ---------------------------------------------------------------------------
// Streaming: ResponsesEventToAnthropicEvents tests
// ---------------------------------------------------------------------------

func TestStreamingTextOnly(t *testing.T) {
	state := NewResponsesEventToAnthropicState()

	// 1. response.created
	events := ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type: "response.created",
		Response: &ResponsesResponse{
			ID:    "resp_1",
			Model: "gpt-5.2",
		},
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "message_start", events[0].Type)

	// 2. output_item.added (message)
	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 0,
		Item:        &ResponsesOutput{Type: "message"},
	}, state)
	assert.Len(t, events, 0) // message item doesn't emit events

	// 3. text delta
	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:  "response.output_text.delta",
		Delta: "Hello",
	}, state)
	require.Len(t, events, 2) // content_block_start + content_block_delta
	assert.Equal(t, "content_block_start", events[0].Type)
	assert.Equal(t, "text", events[0].ContentBlock.Type)
	assert.Equal(t, "content_block_delta", events[1].Type)
	assert.Equal(t, "text_delta", events[1].Delta.Type)
	assert.Equal(t, "Hello", events[1].Delta.Text)

	// 4. more text
	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:  "response.output_text.delta",
		Delta: " world",
	}, state)
	require.Len(t, events, 1) // only delta, no new block start
	assert.Equal(t, "content_block_delta", events[0].Type)

	// 5. text done — does NOT close block (lazy close on next item or completed)
	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type: "response.output_text.done",
	}, state)
	require.Len(t, events, 0)

	// 6. completed — closes open block + message_delta + message_stop
	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type: "response.completed",
		Response: &ResponsesResponse{
			Status: "completed",
			Usage:  &ResponsesUsage{InputTokens: 10, OutputTokens: 5},
		},
	}, state)
	require.Len(t, events, 3) // content_block_stop + message_delta + message_stop
	assert.Equal(t, "content_block_stop", events[0].Type)
	assert.Equal(t, "message_delta", events[1].Type)
	assert.Equal(t, "end_turn", events[1].Delta.StopReason)
	assert.Equal(t, 10, events[1].Usage.InputTokens)
	assert.Equal(t, 5, events[1].Usage.OutputTokens)
	assert.Equal(t, "message_stop", events[2].Type)
}

func TestStreamingCachedTokensUseAnthropicInputSemantics(t *testing.T) {
	state := NewResponsesEventToAnthropicState()
	ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:     "response.created",
		Response: &ResponsesResponse{ID: "resp_cached_stream", Model: "gpt-5.2"},
	}, state)

	events := ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type: "response.completed",
		Response: &ResponsesResponse{
			Status: "completed",
			Usage: &ResponsesUsage{
				InputTokens:  54006,
				OutputTokens: 123,
				TotalTokens:  54129,
				InputTokensDetails: &ResponsesInputTokensDetails{
					CachedTokens: 50688,
				},
			},
		},
	}, state)

	require.Len(t, events, 2)
	assert.Equal(t, "message_delta", events[0].Type)
	assert.Equal(t, 3318, events[0].Usage.InputTokens)
	assert.Equal(t, 50688, events[0].Usage.CacheReadInputTokens)
	assert.Equal(t, 123, events[0].Usage.OutputTokens)
	assert.Equal(t, "message_stop", events[1].Type)
}

func TestStreamingToolCall(t *testing.T) {
	state := NewResponsesEventToAnthropicState()

	// 1. response.created
	ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:     "response.created",
		Response: &ResponsesResponse{ID: "resp_2", Model: "gpt-5.2"},
	}, state)

	// 2. function_call added
	events := ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 0,
		Item:        &ResponsesOutput{Type: "function_call", CallID: "call_1", Name: "get_weather"},
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "content_block_start", events[0].Type)
	assert.Equal(t, "tool_use", events[0].ContentBlock.Type)
	assert.Equal(t, "call_1", events[0].ContentBlock.ID)

	// 3. arguments delta
	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.function_call_arguments.delta",
		OutputIndex: 0,
		Delta:       `{"city":`,
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "content_block_delta", events[0].Type)
	assert.Equal(t, "input_json_delta", events[0].Delta.Type)
	assert.Equal(t, `{"city":`, events[0].Delta.PartialJSON)

	// 4. arguments done — does NOT close block
	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type: "response.function_call_arguments.done",
	}, state)
	require.Len(t, events, 0)

	// 5. output_item.done closes the function_call block
	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.output_item.done",
		OutputIndex: 0,
		Item:        &ResponsesOutput{Type: "function_call", Status: "completed"},
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "content_block_stop", events[0].Type)

	// 6. completed with tool_calls
	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type: "response.completed",
		Response: &ResponsesResponse{
			Status: "completed",
			Usage:  &ResponsesUsage{InputTokens: 20, OutputTokens: 10},
		},
	}, state)
	require.Len(t, events, 2)
	assert.Equal(t, "tool_use", events[0].Delta.StopReason)
}

func TestStreamingReadToolDropsEmptyPages(t *testing.T) {
	state := NewResponsesEventToAnthropicState()

	ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:     "response.created",
		Response: &ResponsesResponse{ID: "resp_read_stream", Model: "gpt-5.5"},
	}, state)

	events := ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 0,
		Item:        &ResponsesOutput{Type: "function_call", CallID: "call_read", Name: "Read"},
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "content_block_start", events[0].Type)

	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.function_call_arguments.delta",
		OutputIndex: 0,
		Delta:       `{"file_path":"/tmp/demo.py","limit":2000,"offset":0,"pages":""}`,
	}, state)
	assert.Len(t, events, 0)

	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.function_call_arguments.done",
		OutputIndex: 0,
		Arguments:   `{"file_path":"/tmp/demo.py","limit":2000,"offset":0,"pages":""}`,
	}, state)
	require.Len(t, events, 2)
	assert.Equal(t, "content_block_delta", events[0].Type)
	assert.Equal(t, "input_json_delta", events[0].Delta.Type)
	assert.JSONEq(t, `{"file_path":"/tmp/demo.py","limit":2000,"offset":0}`, events[0].Delta.PartialJSON)
	assert.Equal(t, "content_block_stop", events[1].Type)
}

func TestStreamingReasoning(t *testing.T) {
	state := NewResponsesEventToAnthropicState()

	ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:     "response.created",
		Response: &ResponsesResponse{ID: "resp_3", Model: "gpt-5.2"},
	}, state)

	// reasoning item added
	events := ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 0,
		Item:        &ResponsesOutput{Type: "reasoning"},
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "content_block_start", events[0].Type)
	assert.Equal(t, "thinking", events[0].ContentBlock.Type)

	// reasoning text delta
	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.reasoning_summary_text.delta",
		OutputIndex: 0,
		Delta:       "Let me think...",
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "content_block_delta", events[0].Type)
	assert.Equal(t, "thinking_delta", events[0].Delta.Type)
	assert.Equal(t, "Let me think...", events[0].Delta.Thinking)

	// reasoning done — does NOT close block
	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type: "response.reasoning_summary_text.done",
	}, state)
	require.Len(t, events, 0)

	// output_item.done for reasoning — emits signature + keeps block open for lazy close
	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.output_item.done",
		OutputIndex: 0,
		Item: &ResponsesOutput{
			Type:             "reasoning",
			ID:               "rs_abc123",
			EncryptedContent: "enc_data",
		},
	}, state)
	// Should emit signature_delta (thinking_delta placeholder skipped because HasReasoningDelta=true)
	require.Len(t, events, 1)
	assert.Equal(t, "content_block_delta", events[0].Type)
	assert.Equal(t, "signature_delta", events[0].Delta.Type)
	require.NotEmpty(t, events[0].Delta.Signature)
	assert.Contains(t, events[0].Delta.Signature, openAIReasoningSignaturePrefix)
	decoded := decodeOpenAIReasoningSignatureEnvelope(events[0].Delta.Signature)
	require.NotNil(t, decoded)
	assert.Equal(t, "reasoning", decoded.Type)
	assert.Equal(t, "enc_data", decoded.EncryptedContent)
	assert.Equal(t, "rs_abc123", decoded.ID)
}

func TestStreamingIncomplete(t *testing.T) {
	state := NewResponsesEventToAnthropicState()

	ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:     "response.created",
		Response: &ResponsesResponse{ID: "resp_4", Model: "gpt-5.2"},
	}, state)

	ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:  "response.output_text.delta",
		Delta: "Partial output...",
	}, state)

	events := ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type: "response.incomplete",
		Response: &ResponsesResponse{
			Status:            "incomplete",
			IncompleteDetails: &ResponsesIncompleteDetails{Reason: "max_output_tokens"},
			Usage:             &ResponsesUsage{InputTokens: 100, OutputTokens: 4096},
		},
	}, state)

	// Should close the text block + message_delta + message_stop
	require.Len(t, events, 3)
	assert.Equal(t, "content_block_stop", events[0].Type)
	assert.Equal(t, "message_delta", events[1].Type)
	assert.Equal(t, "max_tokens", events[1].Delta.StopReason)
	assert.Equal(t, "message_stop", events[2].Type)
}

func TestFinalizeStream_NeverStarted(t *testing.T) {
	state := NewResponsesEventToAnthropicState()
	events := FinalizeResponsesAnthropicStream(state)
	assert.Nil(t, events)
}

func TestFinalizeStream_AlreadyCompleted(t *testing.T) {
	state := NewResponsesEventToAnthropicState()
	state.MessageStartSent = true
	state.MessageStopSent = true
	events := FinalizeResponsesAnthropicStream(state)
	assert.Nil(t, events)
}

func TestFinalizeStream_AbnormalTermination(t *testing.T) {
	state := NewResponsesEventToAnthropicState()

	// Simulate a stream that started but never completed
	ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:     "response.created",
		Response: &ResponsesResponse{ID: "resp_5", Model: "gpt-5.2"},
	}, state)

	ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:  "response.output_text.delta",
		Delta: "Interrupted...",
	}, state)

	// Stream ends without response.completed
	events := FinalizeResponsesAnthropicStream(state)
	require.Len(t, events, 3) // content_block_stop + message_delta + message_stop
	assert.Equal(t, "content_block_stop", events[0].Type)
	assert.Equal(t, "message_delta", events[1].Type)
	assert.Equal(t, "end_turn", events[1].Delta.StopReason)
	assert.Equal(t, "message_stop", events[2].Type)
}

func TestStreamingEmptyResponse(t *testing.T) {
	state := NewResponsesEventToAnthropicState()

	ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:     "response.created",
		Response: &ResponsesResponse{ID: "resp_6", Model: "gpt-5.2"},
	}, state)

	events := ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type: "response.completed",
		Response: &ResponsesResponse{
			Status: "completed",
			Usage:  &ResponsesUsage{InputTokens: 5, OutputTokens: 0},
		},
	}, state)

	require.Len(t, events, 2) // message_delta + message_stop
	assert.Equal(t, "message_delta", events[0].Type)
	assert.Equal(t, "end_turn", events[0].Delta.StopReason)
}

func TestResponsesAnthropicEventToSSE(t *testing.T) {
	evt := AnthropicStreamEvent{
		Type: "message_start",
		Message: &AnthropicResponse{
			ID:   "resp_1",
			Type: "message",
			Role: "assistant",
		},
	}
	sse, err := ResponsesAnthropicEventToSSE(evt)
	require.NoError(t, err)
	assert.Contains(t, sse, "event: message_start\n")
	assert.Contains(t, sse, "data: ")
	assert.Contains(t, sse, `"resp_1"`)
}

// ---------------------------------------------------------------------------
// response.failed tests
// ---------------------------------------------------------------------------

func TestStreamingFailed(t *testing.T) {
	state := NewResponsesEventToAnthropicState()

	// 1. response.created
	ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:     "response.created",
		Response: &ResponsesResponse{ID: "resp_fail_1", Model: "gpt-5.2"},
	}, state)

	// 2. Some text output before failure
	ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:  "response.output_text.delta",
		Delta: "Partial output before failure",
	}, state)

	// 3. response.failed
	events := ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type: "response.failed",
		Response: &ResponsesResponse{
			Status: "failed",
			Error:  &ResponsesError{Code: "server_error", Message: "Internal error"},
			Usage:  &ResponsesUsage{InputTokens: 50, OutputTokens: 10},
		},
	}, state)

	// Should close text block + error event
	require.Len(t, events, 2)
	assert.Equal(t, "content_block_stop", events[0].Type)
	assert.Equal(t, "error", events[1].Type)
	assert.Equal(t, "api_error", events[1].Error.Type)
	assert.Equal(t, "Internal error", events[1].Error.Message)
}

func TestStreamingFailedNoOutput(t *testing.T) {
	state := NewResponsesEventToAnthropicState()

	// 1. response.created
	ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:     "response.created",
		Response: &ResponsesResponse{ID: "resp_fail_2", Model: "gpt-5.2"},
	}, state)

	// 2. response.failed with no prior output
	events := ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type: "response.failed",
		Response: &ResponsesResponse{
			Status: "failed",
			Error:  &ResponsesError{Code: "rate_limit_error", Message: "Too many requests"},
			Usage:  &ResponsesUsage{InputTokens: 20, OutputTokens: 0},
		},
	}, state)

	// Should emit error event (no block to close)
	require.Len(t, events, 1)
	assert.Equal(t, "error", events[0].Type)
	assert.Equal(t, "Too many requests", events[0].Error.Message)
}

func TestResponsesToAnthropic_Failed(t *testing.T) {
	resp := &ResponsesResponse{
		ID:     "resp_fail_3",
		Model:  "gpt-5.2",
		Status: "failed",
		Error:  &ResponsesError{Code: "server_error", Message: "Something went wrong"},
		Output: []ResponsesOutput{},
		Usage:  &ResponsesUsage{InputTokens: 30, OutputTokens: 0},
	}

	anth := ResponsesToAnthropic(resp, "claude-opus-4-6")
	// Failed status defaults to "end_turn" stop reason
	assert.Equal(t, "end_turn", anth.StopReason)
	// Should have at least an empty text block
	require.Len(t, anth.Content, 1)
	assert.Equal(t, "text", anth.Content[0].Type)
}

// ---------------------------------------------------------------------------
// thinking → reasoning conversion tests
// ---------------------------------------------------------------------------

func TestAnthropicToResponses_ThinkingEnabled(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages:  []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		Thinking:  &AnthropicThinking{Type: "enabled", BudgetTokens: 10000},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	require.NotNil(t, resp.Reasoning)
	// thinking.type is ignored for effort; default high applies.
	assert.Equal(t, "high", resp.Reasoning.Effort)
	assert.Equal(t, "detailed", resp.Reasoning.Summary)
	assert.Contains(t, resp.Include, "reasoning.encrypted_content")
	assert.NotContains(t, resp.Include, "reasoning.summary")
}

func TestAnthropicToResponses_ThinkingAdaptive(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages:  []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		Thinking:  &AnthropicThinking{Type: "adaptive", BudgetTokens: 5000},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	require.NotNil(t, resp.Reasoning)
	// thinking.type is ignored for effort; default high applies.
	assert.Equal(t, "high", resp.Reasoning.Effort)
	assert.Equal(t, "detailed", resp.Reasoning.Summary)
	assert.NotContains(t, resp.Include, "reasoning.summary")
}

func TestAnthropicToResponses_ThinkingDisabled(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages:  []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		Thinking:  &AnthropicThinking{Type: "disabled"},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	// Default effort applies (high → high) even when thinking is disabled.
	require.NotNil(t, resp.Reasoning)
	assert.Equal(t, "high", resp.Reasoning.Effort)
}

func TestAnthropicToResponses_NoThinking(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages:  []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	// Default effort applies (high → high) when no thinking/output_config is set.
	require.NotNil(t, resp.Reasoning)
	assert.Equal(t, "high", resp.Reasoning.Effort)
}

// ---------------------------------------------------------------------------
// output_config.effort override tests
// ---------------------------------------------------------------------------

func TestAnthropicToResponses_OutputConfigOverridesDefault(t *testing.T) {
	// Default is high, but output_config.effort="low" overrides. low→low after mapping.
	req := &AnthropicRequest{
		Model:        "gpt-5.2",
		MaxTokens:    1024,
		Messages:     []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		Thinking:     &AnthropicThinking{Type: "enabled", BudgetTokens: 10000},
		OutputConfig: &AnthropicOutputConfig{Effort: "low"},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	require.NotNil(t, resp.Reasoning)
	assert.Equal(t, "low", resp.Reasoning.Effort)
	assert.Equal(t, "detailed", resp.Reasoning.Summary)
}

func TestAnthropicToResponses_OutputConfigWithoutThinking(t *testing.T) {
	// No thinking field, but output_config.effort="medium" → creates reasoning.
	// medium→medium after 1:1 mapping.
	req := &AnthropicRequest{
		Model:        "gpt-5.2",
		MaxTokens:    1024,
		Messages:     []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		OutputConfig: &AnthropicOutputConfig{Effort: "medium"},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	require.NotNil(t, resp.Reasoning)
	assert.Equal(t, "medium", resp.Reasoning.Effort)
	assert.Equal(t, "detailed", resp.Reasoning.Summary)
}

func TestAnthropicEventToResponsesStream_BuildsCompletedOutput(t *testing.T) {
	state := NewAnthropicEventToResponsesState()

	events := AnthropicEventToResponsesEvents(&AnthropicStreamEvent{
		Type: "message_start",
		Message: &AnthropicResponse{
			ID:    "msg_1",
			Model: "claude-opus-4-6",
			Usage: AnthropicUsage{InputTokens: 11},
		},
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "response.created", events[0].Type)

	idx0 := 0
	events = AnthropicEventToResponsesEvents(&AnthropicStreamEvent{
		Type:         "content_block_start",
		Index:        &idx0,
		ContentBlock: &AnthropicContentBlock{Type: "text"},
	}, state)
	require.Len(t, events, 2)
	assert.Equal(t, "response.output_item.added", events[0].Type)
	assert.Equal(t, "response.content_part.added", events[1].Type)
	var part ResponsesOutputPart
	require.NoError(t, json.Unmarshal(events[1].Part, &part))
	assert.Equal(t, "output_text", part.Type)

	events = AnthropicEventToResponsesEvents(&AnthropicStreamEvent{
		Type:  "content_block_delta",
		Index: &idx0,
		Delta: &AnthropicDelta{Type: "text_delta", Text: "Hello"},
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "response.output_text.delta", events[0].Type)
	assert.Equal(t, "Hello", events[0].Delta)

	events = AnthropicEventToResponsesEvents(&AnthropicStreamEvent{
		Type:  "content_block_stop",
		Index: &idx0,
	}, state)
	require.Len(t, events, 2)
	assert.Equal(t, "response.output_text.done", events[0].Type)
	assert.Equal(t, "response.content_part.done", events[1].Type)

	idx1 := 1
	events = AnthropicEventToResponsesEvents(&AnthropicStreamEvent{
		Type:         "content_block_start",
		Index:        &idx1,
		ContentBlock: &AnthropicContentBlock{Type: "thinking"},
	}, state)
	require.Len(t, events, 2)
	assert.Equal(t, "response.output_item.done", events[0].Type)
	assert.Equal(t, "response.output_item.added", events[1].Type)
	assert.Equal(t, "reasoning", events[1].Item.Type)

	events = AnthropicEventToResponsesEvents(&AnthropicStreamEvent{
		Type:  "content_block_delta",
		Index: &idx1,
		Delta: &AnthropicDelta{Type: "thinking_delta", Thinking: "Need to think"},
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "response.reasoning_text.delta", events[0].Type)
	assert.Equal(t, "Need to think", events[0].Delta)

	_ = AnthropicEventToResponsesEvents(&AnthropicStreamEvent{
		Type:  "content_block_delta",
		Index: &idx1,
		Delta: &AnthropicDelta{Type: "signature_delta", Signature: "enc_data@rs_1"},
	}, state)

	events = AnthropicEventToResponsesEvents(&AnthropicStreamEvent{
		Type:  "content_block_stop",
		Index: &idx1,
	}, state)
	require.Len(t, events, 2)
	assert.Equal(t, "response.reasoning_text.done", events[0].Type)
	assert.Equal(t, "response.output_item.done", events[1].Type)

	_ = AnthropicEventToResponsesEvents(&AnthropicStreamEvent{
		Type:  "message_delta",
		Usage: &AnthropicUsage{OutputTokens: 7},
		Delta: &AnthropicDelta{StopReason: "end_turn"},
	}, state)

	events = AnthropicEventToResponsesEvents(&AnthropicStreamEvent{Type: "message_stop"}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "response.completed", events[0].Type)
	require.NotNil(t, events[0].Response)
	assert.Equal(t, "completed", events[0].Response.Status)
	assert.Equal(t, "Hello", events[0].Response.OutputText)
	require.Len(t, events[0].Response.Output, 2)
	assert.Equal(t, "message", events[0].Response.Output[0].Type)
	require.Len(t, events[0].Response.Output[0].Content, 1)
	assert.Equal(t, "Hello", events[0].Response.Output[0].Content[0].Text)
	assert.Equal(t, "reasoning", events[0].Response.Output[1].Type)
	assert.Equal(t, "enc_data", events[0].Response.Output[1].EncryptedContent)
	assert.Equal(t, "rs_1", events[0].Response.Output[1].ID)
	require.NotNil(t, events[0].Response.Usage)
	assert.Equal(t, 11, events[0].Response.Usage.InputTokens)
	assert.Equal(t, 7, events[0].Response.Usage.OutputTokens)
}

func TestResponsesEventToAnthropicEvents_ContentPartRefusalAndReasoningText(t *testing.T) {
	state := NewResponsesEventToAnthropicState()
	ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:     "response.created",
		Response: &ResponsesResponse{ID: "resp_alias", Model: "gpt-5.2"},
	}, state)

	events := ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:         "response.content_part.added",
		OutputIndex:  0,
		ContentIndex: 0,
		Part:         json.RawMessage(`{"type":"output_text","text":"Hello"}`),
	}, state)
	require.Len(t, events, 2)
	assert.Equal(t, "content_block_start", events[0].Type)
	assert.Equal(t, "text", events[0].ContentBlock.Type)
	assert.Equal(t, "content_block_delta", events[1].Type)
	assert.Equal(t, "Hello", events[1].Delta.Text)

	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:  "response.refusal.delta",
		Delta: "Sorry",
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "content_block_delta", events[0].Type)
	assert.Equal(t, "text_delta", events[0].Delta.Type)
	assert.Equal(t, "Sorry", events[0].Delta.Text)

	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 1,
		Item:        &ResponsesOutput{Type: "reasoning"},
	}, state)
	require.Len(t, events, 2)
	assert.Equal(t, "content_block_stop", events[0].Type)
	assert.Equal(t, "content_block_start", events[1].Type)
	assert.Equal(t, "thinking", events[1].ContentBlock.Type)

	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.reasoning_text.delta",
		OutputIndex: 1,
		Delta:       "plan",
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "content_block_delta", events[0].Type)
	assert.Equal(t, "thinking_delta", events[0].Delta.Type)
	assert.Equal(t, "plan", events[0].Delta.Thinking)
}

func TestAnthropicToResponses_OutputConfigHigh(t *testing.T) {
	// output_config.effort="high" → mapped to "high" (1:1, both sides' default).
	req := &AnthropicRequest{
		Model:        "gpt-5.2",
		MaxTokens:    1024,
		Messages:     []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		OutputConfig: &AnthropicOutputConfig{Effort: "high"},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	require.NotNil(t, resp.Reasoning)
	assert.Equal(t, "high", resp.Reasoning.Effort)
	assert.Equal(t, "detailed", resp.Reasoning.Summary)
}

func TestAnthropicToResponses_OutputConfigMax(t *testing.T) {
	// output_config.effort="max" → mapped to OpenAI's highest supported level "xhigh".
	req := &AnthropicRequest{
		Model:        "gpt-5.2",
		MaxTokens:    1024,
		Messages:     []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		OutputConfig: &AnthropicOutputConfig{Effort: "max"},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	require.NotNil(t, resp.Reasoning)
	assert.Equal(t, "xhigh", resp.Reasoning.Effort)
	assert.Equal(t, "detailed", resp.Reasoning.Summary)
}

func TestAnthropicToResponses_NoOutputConfig(t *testing.T) {
	// No output_config → default high regardless of thinking.type.
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages:  []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		Thinking:  &AnthropicThinking{Type: "enabled", BudgetTokens: 10000},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	require.NotNil(t, resp.Reasoning)
	assert.Equal(t, "high", resp.Reasoning.Effort)
}

func TestAnthropicToResponses_OutputConfigWithoutEffort(t *testing.T) {
	req := &AnthropicRequest{
		Model:        "gpt-5.2",
		MaxTokens:    1024,
		Messages:     []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		OutputConfig: &AnthropicOutputConfig{},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	require.NotNil(t, resp.Reasoning)
	assert.Equal(t, "high", resp.Reasoning.Effort)
}

func TestAnthropicToResponses_OutputConfigFormat(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages:  []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		OutputConfig: &AnthropicOutputConfig{
			Format: json.RawMessage(`{"type":"json_schema","name":"weather","schema":{"type":"object"}}`),
		},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	assert.JSONEq(t, `{"format":{"type":"json_schema","name":"weather","schema":{"type":"object"}}}`, string(resp.Text))
	require.NotNil(t, resp.Reasoning)
	assert.Equal(t, "high", resp.Reasoning.Effort)
}

// ---------------------------------------------------------------------------
// tool_choice conversion tests
// ---------------------------------------------------------------------------

func TestAnthropicToResponses_ToolChoiceAuto(t *testing.T) {
	req := &AnthropicRequest{
		Model:      "gpt-5.2",
		MaxTokens:  1024,
		Messages:   []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		ToolChoice: json.RawMessage(`{"type":"auto"}`),
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)

	var tc string
	require.NoError(t, json.Unmarshal(resp.ToolChoice, &tc))
	assert.Equal(t, "auto", tc)
}

func TestAnthropicToResponses_ToolChoiceAutoDisableParallel(t *testing.T) {
	req := &AnthropicRequest{
		Model:      "gpt-5.2",
		MaxTokens:  1024,
		Messages:   []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		ToolChoice: json.RawMessage(`{"type":"auto","disable_parallel_tool_use":true}`),
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)

	var tc string
	require.NoError(t, json.Unmarshal(resp.ToolChoice, &tc))
	assert.Equal(t, "auto", tc)
	require.NotNil(t, resp.ParallelToolCalls)
	assert.False(t, *resp.ParallelToolCalls)
}

func TestAnthropicToResponses_ToolChoiceAny(t *testing.T) {
	req := &AnthropicRequest{
		Model:      "gpt-5.2",
		MaxTokens:  1024,
		Messages:   []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		ToolChoice: json.RawMessage(`{"type":"any"}`),
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)

	var tc string
	require.NoError(t, json.Unmarshal(resp.ToolChoice, &tc))
	assert.Equal(t, "required", tc)
}

func TestAnthropicToResponses_ToolChoiceSpecific(t *testing.T) {
	req := &AnthropicRequest{
		Model:      "gpt-5.2",
		MaxTokens:  1024,
		Messages:   []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		ToolChoice: json.RawMessage(`{"type":"tool","name":"get_weather"}`),
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)

	var tc map[string]any
	require.NoError(t, json.Unmarshal(resp.ToolChoice, &tc))
	assert.Equal(t, "function", tc["type"])
	fn, ok := tc["function"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "get_weather", fn["name"])
}

// ---------------------------------------------------------------------------
// Image content block conversion tests
// ---------------------------------------------------------------------------

func TestAnthropicToResponses_UserImageBlock(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages: []AnthropicMessage{
			{Role: "user", Content: json.RawMessage(`[
				{"type":"text","text":"What is in this image?"},
				{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBOR"}}
			]`)},
		},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)

	var items []ResponsesInputItem
	require.NoError(t, json.Unmarshal(resp.Input, &items))
	require.Len(t, items, 1)
	assert.Equal(t, "user", items[0].Role)

	var parts []ResponsesContentPart
	require.NoError(t, json.Unmarshal(items[0].Content, &parts))
	require.Len(t, parts, 2)
	assert.Equal(t, "input_text", parts[0].Type)
	assert.Equal(t, "What is in this image?", parts[0].Text)
	assert.Equal(t, "input_image", parts[1].Type)
	assert.Equal(t, "data:image/png;base64,iVBOR", parts[1].ImageURL)
}

func TestAnthropicToResponses_ImageOnlyUserMessage(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages: []AnthropicMessage{
			{Role: "user", Content: json.RawMessage(`[
				{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"/9j/4AAQ"}}
			]`)},
		},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)

	var items []ResponsesInputItem
	require.NoError(t, json.Unmarshal(resp.Input, &items))
	require.Len(t, items, 1)

	var parts []ResponsesContentPart
	require.NoError(t, json.Unmarshal(items[0].Content, &parts))
	require.Len(t, parts, 1)
	assert.Equal(t, "input_image", parts[0].Type)
	assert.Equal(t, "data:image/jpeg;base64,/9j/4AAQ", parts[0].ImageURL)
}

func TestAnthropicToResponses_ToolResultWithImage(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages: []AnthropicMessage{
			{Role: "user", Content: json.RawMessage(`"Read the screenshot"`)},
			{Role: "assistant", Content: json.RawMessage(`[{"type":"tool_use","id":"toolu_1","name":"Read","input":{"file_path":"/tmp/screen.png"}}]`)},
			{Role: "user", Content: json.RawMessage(`[
				{"type":"tool_result","tool_use_id":"toolu_1","content":[
					{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBOR"}}
				]}
			]`)},
		},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)

	var items []ResponsesInputItem
	require.NoError(t, json.Unmarshal(resp.Input, &items))
	// user + function_call + function_call_output + user(image) = 4
	require.Len(t, items, 4)

	// function_call_output should have text-only output (no image).
	assert.Equal(t, "function_call_output", items[2].Type)
	assert.Equal(t, "toolu_1", items[2].CallID)
	assert.Equal(t, "(empty)", items[2].Output)

	// Image should be in a separate user message.
	assert.Equal(t, "user", items[3].Role)
	var parts []ResponsesContentPart
	require.NoError(t, json.Unmarshal(items[3].Content, &parts))
	require.Len(t, parts, 1)
	assert.Equal(t, "input_image", parts[0].Type)
	assert.Equal(t, "data:image/png;base64,iVBOR", parts[0].ImageURL)
}

func TestAnthropicToResponses_ToolResultMixed(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages: []AnthropicMessage{
			{Role: "user", Content: json.RawMessage(`"Describe the file"`)},
			{Role: "assistant", Content: json.RawMessage(`[{"type":"tool_use","id":"toolu_2","name":"Read","input":{"file_path":"/tmp/photo.png"}}]`)},
			{Role: "user", Content: json.RawMessage(`[
				{"type":"tool_result","tool_use_id":"toolu_2","content":[
					{"type":"text","text":"File metadata: 800x600 PNG"},
					{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}
				]}
			]`)},
		},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)

	var items []ResponsesInputItem
	require.NoError(t, json.Unmarshal(resp.Input, &items))
	// user + function_call + function_call_output + user(image) = 4
	require.Len(t, items, 4)

	// function_call_output should have text-only output.
	assert.Equal(t, "function_call_output", items[2].Type)
	assert.Equal(t, "File metadata: 800x600 PNG", items[2].Output)

	// Image should be in a separate user message.
	assert.Equal(t, "user", items[3].Role)
	var parts []ResponsesContentPart
	require.NoError(t, json.Unmarshal(items[3].Content, &parts))
	require.Len(t, parts, 1)
	assert.Equal(t, "input_image", parts[0].Type)
	assert.Equal(t, "data:image/png;base64,AAAA", parts[0].ImageURL)
}

func TestAnthropicToResponses_TextOnlyToolResultBackwardCompat(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages: []AnthropicMessage{
			{Role: "user", Content: json.RawMessage(`"Check weather"`)},
			{Role: "assistant", Content: json.RawMessage(`[{"type":"tool_use","id":"call_1","name":"get_weather","input":{"city":"NYC"}}]`)},
			{Role: "user", Content: json.RawMessage(`[
				{"type":"tool_result","tool_use_id":"call_1","content":[
					{"type":"text","text":"Sunny, 72°F"}
				]}
			]`)},
		},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)

	var items []ResponsesInputItem
	require.NoError(t, json.Unmarshal(resp.Input, &items))
	// user + function_call + function_call_output = 3
	require.Len(t, items, 3)

	// Text-only tool_result should produce a plain string.
	assert.Equal(t, "Sunny, 72°F", items[2].Output)
}

func TestAnthropicToResponses_ImageEmptyMediaType(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages: []AnthropicMessage{
			{Role: "user", Content: json.RawMessage(`[
				{"type":"image","source":{"type":"base64","media_type":"","data":"iVBOR"}}
			]`)},
		},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)

	var items []ResponsesInputItem
	require.NoError(t, json.Unmarshal(resp.Input, &items))
	require.Len(t, items, 1)

	var parts []ResponsesContentPart
	require.NoError(t, json.Unmarshal(items[0].Content, &parts))
	require.Len(t, parts, 1)
	assert.Equal(t, "input_image", parts[0].Type)
	// Should default to image/png when media_type is empty.
	assert.Equal(t, "data:image/png;base64,iVBOR", parts[0].ImageURL)
}

func TestDecodeCompactionSignature_PreservesEncryptedContentWithAtSigns(t *testing.T) {
	encryptedContent := "enc@part1@@part2"
	id := "rs_123"

	decoded := decodeCompactionSignature(encodeCompactionSignature(id, encryptedContent))
	require.NotNil(t, decoded)
	assert.Equal(t, encryptedContent, decoded.encryptedContent)
	assert.Equal(t, id, decoded.id)
}

func TestOpenAIReasoningSignatureEnvelope_RoundTripsReasoning(t *testing.T) {
	item := ResponsesOutput{
		Type:             "reasoning",
		ID:               "rs_123",
		EncryptedContent: "enc@part1#part2",
		Summary:          []ResponsesSummary{{Type: "summary_text", Text: "kept summary"}},
		Status:           "completed",
	}

	decoded := decodeOpenAIReasoningSignatureEnvelope(encodeReasoningItemSignature(item))
	require.NotNil(t, decoded)
	assert.Equal(t, openAIReasoningSignatureVersion, decoded.Version)
	assert.Equal(t, "reasoning", decoded.Type)
	assert.Equal(t, item.ID, decoded.ID)
	assert.Equal(t, item.EncryptedContent, decoded.EncryptedContent)
	assert.Equal(t, item.Summary, decoded.Summary)
	assert.Equal(t, item.Status, decoded.Status)
}

func TestOpenAIReasoningSignatureEnvelope_RoundTripsCompaction(t *testing.T) {
	item := ResponsesOutput{
		Type:             "compaction",
		ID:               "rs_compact",
		EncryptedContent: "enc@compact#payload",
		Summary:          []ResponsesSummary{{Type: "summary_text", Text: "compact summary"}},
		Status:           "completed",
	}

	decoded := decodeOpenAIReasoningSignatureEnvelope(encodeCompactionItemSignature(item))
	require.NotNil(t, decoded)
	assert.Equal(t, "compaction", decoded.Type)
	assert.Equal(t, item.ID, decoded.ID)
	assert.Equal(t, item.EncryptedContent, decoded.EncryptedContent)
	assert.Equal(t, item.Summary, decoded.Summary)
	assert.Equal(t, item.Status, decoded.Status)
}

func TestResponsesAnthropicReasoningEnvelope_RoundTripPreservesFields(t *testing.T) {
	original := &ResponsesResponse{
		ID:     "resp_reasoning_roundtrip",
		Model:  "gpt-5.5",
		Status: "completed",
		Output: []ResponsesOutput{
			{
				Type:             "reasoning",
				ID:               "rs_roundtrip",
				EncryptedContent: "enc@roundtrip#payload",
				Summary:          []ResponsesSummary{{Type: "summary_text", Text: "roundtrip summary"}},
				Status:           "completed",
			},
			{
				Type:    "message",
				Content: []ResponsesContentPart{{Type: "output_text", Text: "answer"}},
			},
		},
	}

	anthropic := ResponsesToAnthropic(original, "claude-opus-4-6")
	require.Len(t, anthropic.Content, 2)
	assert.Equal(t, "thinking", anthropic.Content[0].Type)
	assert.Contains(t, anthropic.Content[0].Signature, openAIReasoningSignaturePrefix)

	converted, err := AnthropicToResponses(&AnthropicRequest{
		Model:     "gpt-5.5",
		MaxTokens: 1024,
		Messages: []AnthropicMessage{{
			Role:    "assistant",
			Content: mustMarshalJSON(anthropic.Content),
		}},
	})
	require.NoError(t, err)

	var items []ResponsesInputItem
	require.NoError(t, json.Unmarshal(converted.Input, &items))
	require.Len(t, items, 2)
	assert.Equal(t, "reasoning", items[0].Type)
	assert.Equal(t, "rs_roundtrip", items[0].ID)
	assert.Equal(t, "enc@roundtrip#payload", items[0].EncryptedContent)
	assert.Equal(t, []ResponsesSummary{{Type: "summary_text", Text: "roundtrip summary"}}, items[0].Summary)
	assert.Equal(t, "completed", items[0].Status)
}

func TestNormalizeToolParameters(t *testing.T) {
	tests := []struct {
		name     string
		input    json.RawMessage
		expected string
	}{
		{
			name:     "nil input",
			input:    nil,
			expected: `{"type":"object","properties":{}}`,
		},
		{
			name:     "empty input",
			input:    json.RawMessage(``),
			expected: `{"type":"object","properties":{}}`,
		},
		{
			name:     "null input",
			input:    json.RawMessage(`null`),
			expected: `{"type":"object","properties":{}}`,
		},
		{
			name:     "object without properties",
			input:    json.RawMessage(`{"type":"object"}`),
			expected: `{"type":"object","properties":{}}`,
		},
		{
			name:     "object with properties",
			input:    json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			expected: `{"type":"object","properties":{"city":{"type":"string"}}}`,
		},
		{
			name:     "non-object type",
			input:    json.RawMessage(`{"type":"string"}`),
			expected: `{"type":"string"}`,
		},
		{
			name:     "object with additional fields preserved",
			input:    json.RawMessage(`{"type":"object","required":["name"]}`),
			expected: `{"type":"object","required":["name"],"properties":{}}`,
		},
		{
			name:     "invalid JSON passthrough",
			input:    json.RawMessage(`not json`),
			expected: `not json`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := normalizeToolParameters(tt.input)
			if tt.name == "invalid JSON passthrough" {
				assert.Equal(t, tt.expected, string(result))
			} else {
				assert.JSONEq(t, tt.expected, string(result))
			}
		})
	}
}

func TestAnthropicToResponses_ToolWithoutProperties(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages: []AnthropicMessage{
			{Role: "user", Content: json.RawMessage(`"Hello"`)},
		},
		Tools: []AnthropicTool{
			{Name: "mcp__pencil__get_style_guide_tags", Description: "Get style tags", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)

	require.Len(t, resp.Tools, 1)
	assert.Equal(t, "function", resp.Tools[0].Type)
	assert.Equal(t, "mcp__pencil__get_style_guide_tags", resp.Tools[0].Name)

	// Parameters must have "properties" field after normalization.
	var params map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(resp.Tools[0].Parameters, &params))
	assert.Contains(t, params, "properties")
}

func TestAnthropicToResponses_ToolWithNilSchema(t *testing.T) {
	req := &AnthropicRequest{
		Model:     "gpt-5.2",
		MaxTokens: 1024,
		Messages: []AnthropicMessage{
			{Role: "user", Content: json.RawMessage(`"Hello"`)},
		},
		Tools: []AnthropicTool{
			{Name: "simple_tool", Description: "A tool"},
		},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)

	require.Len(t, resp.Tools, 1)
	var params map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(resp.Tools[0].Parameters, &params))
	assert.JSONEq(t, `"object"`, string(params["type"]))
	assert.JSONEq(t, `{}`, string(params["properties"]))
}

func TestAnthropicToResponses_MCPRoundTripFields(t *testing.T) {
	req := &AnthropicRequest{
		Model:             "gpt-5.2",
		MaxTokens:         1024,
		Container:         json.RawMessage(`{"id":"conv_req"}`),
		ContextManagement: json.RawMessage(`{"edits":[{"type":"clear_function_results"}]}`),
		MCPServers: []AnthropicMCPServer{{
			Type:               "url",
			URL:                "https://mcp.example.com",
			Name:               "docs",
			AuthorizationToken: "Bearer test",
		}},
		Tools: []AnthropicTool{{
			Type:          "mcp_toolset",
			MCPServerName: "docs",
			DefaultConfig: json.RawMessage(`{"enabled":false}`),
			Configs:       json.RawMessage(`{"lookup":{"enabled":true}}`),
		}},
		Messages: []AnthropicMessage{
			{Role: "user", Content: json.RawMessage(`"Check docs"`)},
			{Role: "assistant", Content: json.RawMessage(`[
				{"type":"mcp_tool_use","id":"call_mcp","name":"lookup","server_name":"docs","input":{"q":"api"}}
			]`)},
			{Role: "user", Content: json.RawMessage(`[
				{"type":"mcp_tool_result","tool_use_id":"call_mcp","content":[{"type":"text","text":"found"}]}
			]`)},
		},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	assert.JSONEq(t, `{"id":"conv_req"}`, string(resp.Conversation))
	assert.JSONEq(t, `[{"type":"clear_function_results"}]`, string(resp.ContextManagement))

	require.Len(t, resp.Tools, 1)
	assert.Equal(t, "mcp", resp.Tools[0].Type)
	assert.Equal(t, "docs", resp.Tools[0].ServerLabel)
	assert.Equal(t, "https://mcp.example.com", resp.Tools[0].ServerURL)
	assert.Equal(t, "Bearer test", resp.Tools[0].Authorization)
	var params map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(resp.Tools[0].Parameters, &params))
	assert.JSONEq(t, `{"enabled":false}`, string(params["default_config"]))
	assert.JSONEq(t, `{"lookup":{"enabled":true}}`, string(params["configs"]))

	var allowed []string
	require.NoError(t, json.Unmarshal(resp.Tools[0].AllowedTools, &allowed))
	require.Equal(t, []string{"lookup"}, allowed)

	var items []ResponsesInputItem
	require.NoError(t, json.Unmarshal(resp.Input, &items))
	require.Len(t, items, 3)
	assert.Equal(t, "function_call", items[1].Type)
	assert.Equal(t, "docs", items[1].Namespace)
	assert.JSONEq(t, `{"type":"mcp_tool_use","id":"call_mcp","name":"lookup","server_name":"docs","input":{"q":"api"}}`, stripEmptyAnthropicFields(items[1].RawItem))
	assert.Equal(t, "function_call_output", items[2].Type)
	assert.Equal(t, "mcp", items[2].Namespace)
	assert.JSONEq(t, `[{"type":"text","text":"found"}]`, string(items[2].OutputRaw))
	assert.JSONEq(t, `{"type":"mcp_tool_result","tool_use_id":"call_mcp","content":[{"type":"text","text":"found"}]}`, stripEmptyAnthropicFields(items[2].RawItem))
}

func TestResponsesToAnthropicRequest_MCPRoundTripFields(t *testing.T) {
	req := &ResponsesRequest{
		Model: "gpt-5.2",
		Input: json.RawMessage(`[
			{"role":"user","content":"Check docs"},
			{"type":"function_call","call_id":"call_mcp","name":"lookup","arguments":"{\"q\":\"api\"}","status":"completed","namespace":"docs","item":{"type":"mcp_tool_use","id":"call_mcp","name":"lookup","server_name":"docs","input":{"q":"api"}}},
			{"type":"function_call_output","call_id":"call_mcp","output":"found","output_raw":[{"type":"text","text":"found"}],"status":"completed","namespace":"docs","item":{"type":"mcp_tool_result","tool_use_id":"call_mcp","content":[{"type":"text","text":"found"}]}}
		]`),
		Conversation:      json.RawMessage(`{"id":"conv_req"}`),
		ContextManagement: json.RawMessage(`[{"type":"clear_function_results"}]`),
		Tools: []ResponsesTool{{
			Type:          "mcp",
			ServerLabel:   "docs",
			ServerURL:     "https://mcp.example.com",
			Authorization: "Bearer test",
			Parameters:    json.RawMessage(`{"default_config":{"enabled":false},"configs":{"lookup":{"enabled":true}}}`),
		}},
	}

	anth, err := ResponsesToAnthropicRequest(req)
	require.NoError(t, err)
	assert.JSONEq(t, `{"id":"conv_req"}`, string(anth.Container))
	assert.JSONEq(t, `{"edits":[{"type":"clear_function_results"}]}`, string(anth.ContextManagement))
	require.Len(t, anth.MCPServers, 1)
	assert.Equal(t, "docs", anth.MCPServers[0].Name)
	assert.Equal(t, "https://mcp.example.com", anth.MCPServers[0].URL)
	assert.Equal(t, "Bearer test", anth.MCPServers[0].AuthorizationToken)
	require.Len(t, anth.Tools, 1)
	assert.Equal(t, "mcp_toolset", anth.Tools[0].Type)
	assert.Equal(t, "docs", anth.Tools[0].MCPServerName)
	assert.JSONEq(t, `{"enabled":false}`, string(anth.Tools[0].DefaultConfig))
	assert.JSONEq(t, `{"lookup":{"enabled":true}}`, string(anth.Tools[0].Configs))

	require.Len(t, anth.Messages, 3)
	var assistantBlocks []AnthropicContentBlock
	require.NoError(t, json.Unmarshal(anth.Messages[1].Content, &assistantBlocks))
	require.Len(t, assistantBlocks, 1)
	assert.Equal(t, "mcp_tool_use", assistantBlocks[0].Type)
	assert.Equal(t, "docs", assistantBlocks[0].ServerName)

	var userBlocks []AnthropicContentBlock
	require.NoError(t, json.Unmarshal(anth.Messages[2].Content, &userBlocks))
	require.Len(t, userBlocks, 1)
	assert.Equal(t, "mcp_tool_result", userBlocks[0].Type)
	assert.JSONEq(t, `[{"type":"text","text":"found"}]`, string(userBlocks[0].Content))
}

func TestAnthropicToResponses_ContextManagementCompaction(t *testing.T) {
	req := &AnthropicRequest{
		Model:             "gpt-5.2",
		MaxTokens:         1024,
		Messages:          []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		ContextManagement: json.RawMessage(`{"edits":[{"type":"compact_20260112","trigger":{"type":"input_tokens","value":150000}},{"type":"clear_function_results"}]}`),
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	assert.JSONEq(t, `[{"type":"compaction","compact_threshold":150000},{"type":"clear_function_results"}]`, string(resp.ContextManagement))
}

func TestResponsesToAnthropicRequest_TextFormatAndEffort(t *testing.T) {
	text := json.RawMessage(`{"format":{"type":"json_schema","name":"weather","schema":{"type":"object"}}}`)
	maxTokens := 512
	req := &ResponsesRequest{
		Model:           "gpt-5.2",
		Input:           json.RawMessage(`[{"role":"user","content":"Hello"}]`),
		MaxOutputTokens: &maxTokens,
		Text:            text,
		Reasoning:       &ResponsesReasoning{Effort: "medium"},
	}

	anth, err := ResponsesToAnthropicRequest(req)
	require.NoError(t, err)
	require.NotNil(t, anth.OutputConfig)
	assert.Equal(t, "medium", anth.OutputConfig.Effort)
	assert.JSONEq(t, `{"type":"json_schema","name":"weather","schema":{"type":"object"}}`, string(anth.OutputConfig.Format))
}

func TestResponsesToAnthropicRequest_DisablesParallelToolCalls(t *testing.T) {
	parallelToolCalls := false
	req := &ResponsesRequest{
		Model:             "gpt-5.2",
		Input:             json.RawMessage(`[{"role":"user","content":"Hello"}]`),
		ParallelToolCalls: &parallelToolCalls,
		ToolChoice:        json.RawMessage(`"required"`),
	}

	anth, err := ResponsesToAnthropicRequest(req)
	require.NoError(t, err)
	assert.JSONEq(t, `{"type":"any","disable_parallel_tool_use":true}`, string(anth.ToolChoice))
}

func TestResponsesToAnthropicRequest_NoneReasoningDisablesThinking(t *testing.T) {
	req := &ResponsesRequest{
		Model:     "gpt-5.2",
		Input:     json.RawMessage(`[{"role":"user","content":"Hello"}]`),
		Reasoning: &ResponsesReasoning{Effort: "minimal"},
	}

	anth, err := ResponsesToAnthropicRequest(req)
	require.NoError(t, err)
	assert.Nil(t, anth.Thinking)
	assert.Nil(t, anth.OutputConfig)
}

func TestResponsesToAnthropicRequest_FileAndAudioFallback(t *testing.T) {
	req := &ResponsesRequest{
		Model: "gpt-5.2",
		Input: json.RawMessage(`[
			{"role":"user","content":[
				{"type":"input_audio","input_audio":{"format":"wav","data":"AAA"}},
				{"type":"input_file","file_id":"file_123","filename":"report.pdf"}
			]}
		]`),
	}

	anth, err := ResponsesToAnthropicRequest(req)
	require.NoError(t, err)
	require.Len(t, anth.Messages, 1)
	var blocks []AnthropicContentBlock
	require.NoError(t, json.Unmarshal(anth.Messages[0].Content, &blocks))
	require.Len(t, blocks, 2)
	assert.Equal(t, "text", blocks[0].Type)
	assert.Equal(t, "[audio input omitted in Anthropic conversion: format=wav]", blocks[0].Text)
	assert.Equal(t, "text", blocks[1].Type)
	assert.Equal(t, "[file input omitted in Anthropic conversion: file_id=file_123, filename=report.pdf]", blocks[1].Text)
}

func TestResponsesToAnthropicRequest_ContextManagementCompaction(t *testing.T) {
	req := &ResponsesRequest{
		Model:             "gpt-5.2",
		Input:             json.RawMessage(`[{"role":"user","content":"Hello"}]`),
		ContextManagement: json.RawMessage(`[{"type":"compaction","compact_threshold":150000},{"type":"clear_function_results"}]`),
	}

	anth, err := ResponsesToAnthropicRequest(req)
	require.NoError(t, err)
	assert.JSONEq(t, `{"edits":[{"type":"compact_20260112","trigger":{"type":"input_tokens","value":150000}},{"type":"clear_function_results"}]}`, string(anth.ContextManagement))
}

func TestAnthropicToResponsesResponse_MCPToolUseAndContainer(t *testing.T) {
	resp := &AnthropicResponse{
		ID:        "msg_mcp",
		Type:      "message",
		Role:      "assistant",
		Model:     "claude-opus-4-6",
		Container: json.RawMessage(`{"id":"conv_resp"}`),
		Content: []AnthropicContentBlock{{
			Type:       "mcp_tool_use",
			ID:         "call_mcp",
			Name:       "lookup",
			ServerName: "docs",
			Input:      json.RawMessage(`{"q":"api"}`),
		}},
		StopReason: "tool_use",
	}

	out := AnthropicToResponsesResponse(resp)
	require.NotNil(t, out.Conversation)
	assert.Equal(t, "conv_resp", out.Conversation.ID)
	require.Len(t, out.Output, 1)
	assert.Equal(t, "function_call", out.Output[0].Type)
	assert.Equal(t, "docs", out.Output[0].Namespace)
	assert.JSONEq(t, `{"type":"mcp_tool_use","id":"call_mcp","name":"lookup","server_name":"docs","input":{"q":"api"}}`, stripEmptyAnthropicFields(out.Output[0].RawItem))
}

func TestResponsesToAnthropic_MCPToolUseAndConversation(t *testing.T) {
	resp := &ResponsesResponse{
		ID:           "resp_mcp",
		Model:        "gpt-5.2",
		Status:       "completed",
		Conversation: &ResponsesConversation{ID: "conv_resp"},
		Output: []ResponsesOutput{{
			Type:      "function_call",
			CallID:    "call_mcp",
			Name:      "lookup",
			Arguments: `{"q":"api"}`,
			Namespace: "docs",
			RawItem:   json.RawMessage(`{"type":"mcp_tool_use","id":"call_mcp","name":"lookup","server_name":"docs","input":{"q":"api"}}`),
		}},
	}

	anth := ResponsesToAnthropic(resp, "claude-opus-4-6")
	assert.JSONEq(t, `{"id":"conv_resp"}`, string(anth.Container))
	require.Len(t, anth.Content, 1)
	assert.Equal(t, "mcp_tool_use", anth.Content[0].Type)
	assert.Equal(t, "docs", anth.Content[0].ServerName)
	assert.Equal(t, "tool_use", anth.StopReason)
}

func TestAnthropicEventToResponsesEvents_MCPToolUse(t *testing.T) {
	state := NewAnthropicEventToResponsesState()

	events := AnthropicEventToResponsesEvents(&AnthropicStreamEvent{
		Type: "message_start",
		Message: &AnthropicResponse{
			ID:    "msg_stream_mcp",
			Model: "claude-opus-4-6",
		},
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "response.created", events[0].Type)

	idx := 0
	events = AnthropicEventToResponsesEvents(&AnthropicStreamEvent{
		Type:  "content_block_start",
		Index: &idx,
		ContentBlock: &AnthropicContentBlock{
			Type:       "mcp_tool_use",
			ID:         "call_mcp",
			Name:       "lookup",
			ServerName: "docs",
		},
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "response.output_item.added", events[0].Type)
	require.NotNil(t, events[0].Item)
	assert.Equal(t, "docs", events[0].Item.Namespace)
	assert.JSONEq(t, `{"type":"mcp_tool_use","id":"call_mcp","name":"lookup","server_name":"docs"}`, stripEmptyAnthropicFields(events[0].Item.RawItem))

	events = AnthropicEventToResponsesEvents(&AnthropicStreamEvent{
		Type:  "content_block_delta",
		Index: &idx,
		Delta: &AnthropicDelta{Type: "input_json_delta", PartialJSON: `{"q":"api"}`},
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "response.function_call_arguments.delta", events[0].Type)

	events = AnthropicEventToResponsesEvents(&AnthropicStreamEvent{Type: "content_block_stop", Index: &idx}, state)
	require.Len(t, events, 2)
	assert.Equal(t, "response.function_call_arguments.done", events[0].Type)
	assert.Equal(t, "response.output_item.done", events[1].Type)
	require.NotNil(t, events[1].Item)
	assert.Equal(t, "docs", events[1].Item.Namespace)

	_ = AnthropicEventToResponsesEvents(&AnthropicStreamEvent{
		Type:  "message_delta",
		Delta: &AnthropicDelta{StopReason: "tool_use"},
		Usage: &AnthropicUsage{OutputTokens: 3},
	}, state)
	events = AnthropicEventToResponsesEvents(&AnthropicStreamEvent{Type: "message_stop"}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "response.completed", events[0].Type)
	require.NotNil(t, events[0].Response)
	require.Len(t, events[0].Response.Output, 1)
	assert.Equal(t, "docs", events[0].Response.Output[0].Namespace)
}

func TestResponsesEventToAnthropicEvents_MCPToolUse(t *testing.T) {
	state := NewResponsesEventToAnthropicState()

	events := ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:     "response.created",
		Response: &ResponsesResponse{ID: "resp_stream_mcp", Model: "gpt-5.2"},
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "message_start", events[0].Type)

	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 0,
		Item: &ResponsesOutput{
			Type:      "function_call",
			CallID:    "call_mcp",
			Name:      "lookup",
			Namespace: "docs",
			RawItem:   json.RawMessage(`{"type":"mcp_tool_use","id":"call_mcp","name":"lookup","server_name":"docs"}`),
		},
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "content_block_start", events[0].Type)
	require.NotNil(t, events[0].ContentBlock)
	assert.Equal(t, "mcp_tool_use", events[0].ContentBlock.Type)
	assert.Equal(t, "docs", events[0].ContentBlock.ServerName)

	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.function_call_arguments.delta",
		OutputIndex: 0,
		Delta:       `{"q":"api"}`,
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "content_block_delta", events[0].Type)
	assert.Equal(t, "input_json_delta", events[0].Delta.Type)

	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type:        "response.output_item.done",
		OutputIndex: 0,
		Item:        &ResponsesOutput{Type: "function_call", Status: "completed"},
	}, state)
	require.Len(t, events, 1)
	assert.Equal(t, "content_block_stop", events[0].Type)

	events = ResponsesEventToAnthropicEvents(&ResponsesStreamEvent{
		Type: "response.completed",
		Response: &ResponsesResponse{
			Status: "completed",
			Usage:  &ResponsesUsage{InputTokens: 7, OutputTokens: 3},
		},
	}, state)
	require.Len(t, events, 2)
	assert.Equal(t, "message_delta", events[0].Type)
	assert.Equal(t, "tool_use", events[0].Delta.StopReason)
	assert.Equal(t, "message_stop", events[1].Type)
}

func TestAnthropicToResponses_UnknownToolChoiceFiltered(t *testing.T) {
	req := &AnthropicRequest{
		Model:      "gpt-5.2",
		MaxTokens:  1024,
		Messages:   []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		ToolChoice: json.RawMessage(`{"type":"foobar","name":"x"}`),
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	require.Len(t, resp.ToolChoice, 0)
}

func TestAnthropicToResponses_UnknownEffortFallsBackToHigh(t *testing.T) {
	req := &AnthropicRequest{
		Model:        "gpt-5.2",
		MaxTokens:    1024,
		Messages:     []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
		OutputConfig: &AnthropicOutputConfig{Effort: "surprising"},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	require.NotNil(t, resp.Reasoning)
	require.Equal(t, "high", resp.Reasoning.Effort)
}

func TestAnthropicToResponses_ContextManagementEditsArray(t *testing.T) {
	req := &AnthropicRequest{
		Model:             "gpt-5.2",
		MaxTokens:         1024,
		ContextManagement: json.RawMessage(`{"edits":[{"type":"clear_function_results"}]}`),
		Messages:          []AnthropicMessage{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
	}

	resp, err := AnthropicToResponses(req)
	require.NoError(t, err)
	assert.JSONEq(t, `[{"type":"clear_function_results"}]`, string(resp.ContextManagement))
}

func TestResponsesToAnthropicRequest_ContextManagementWrapsEditsArray(t *testing.T) {
	req := &ResponsesRequest{
		Model:             "gpt-5.2",
		Input:             json.RawMessage(`[{"role":"user","content":"Hello"}]`),
		ContextManagement: json.RawMessage(`[{"type":"clear_function_results"}]`),
	}

	anth, err := ResponsesToAnthropicRequest(req)
	require.NoError(t, err)
	assert.JSONEq(t, `{"edits":[{"type":"clear_function_results"}]}`, string(anth.ContextManagement))
}
