// Package apicompat provides type definitions and conversion utilities for
// translating between Anthropic Messages and OpenAI Responses API formats.
// It enables multi-protocol support so that clients using different API
// formats can be served through a unified gateway.
package apicompat

import "encoding/json"

// ---------------------------------------------------------------------------
// Anthropic Messages API types
// ---------------------------------------------------------------------------

// AnthropicRequest is the request body for POST /v1/messages.
type AnthropicRequest struct {
	Model             string                 `json:"model"`
	MaxTokens         int                    `json:"max_tokens"`
	System            json.RawMessage        `json:"system,omitempty"` // string or []AnthropicContentBlock
	Messages          []AnthropicMessage     `json:"messages"`
	Tools             []AnthropicTool        `json:"tools,omitempty"`
	MCPServers        []AnthropicMCPServer   `json:"mcp_servers,omitempty"`
	Container         json.RawMessage        `json:"container,omitempty"`
	ContextManagement json.RawMessage        `json:"context_management,omitempty"`
	Stream            bool                   `json:"stream,omitempty"`
	Temperature       *float64               `json:"temperature,omitempty"`
	TopP              *float64               `json:"top_p,omitempty"`
	StopSeqs          []string               `json:"stop_sequences,omitempty"`
	Thinking          *AnthropicThinking     `json:"thinking,omitempty"`
	ToolChoice        json.RawMessage        `json:"tool_choice,omitempty"`
	OutputConfig      *AnthropicOutputConfig `json:"output_config,omitempty"`
	ServiceTier       string                 `json:"service_tier,omitempty"`
	Metadata          json.RawMessage        `json:"metadata,omitempty"`
}

// AnthropicOutputConfig controls output generation parameters.
type AnthropicOutputConfig struct {
	Effort string          `json:"effort,omitempty"` // "low" | "medium" | "high" | "max"
	Format json.RawMessage `json:"format,omitempty"`
}

// AnthropicThinking configures extended thinking in the Anthropic API.
type AnthropicThinking struct {
	Type         string `json:"type"`                    // "enabled" | "adaptive" | "disabled"
	BudgetTokens int    `json:"budget_tokens,omitempty"` // max thinking tokens
}

// AnthropicMessage is a single message in the Anthropic conversation.
type AnthropicMessage struct {
	Role    string          `json:"role"` // "user" | "assistant"
	Content json.RawMessage `json:"content"`
}

// AnthropicContentBlock is one block inside a message's content array.
type AnthropicContentBlock struct {
	Type string `json:"type"`

	// type=text
	Text string `json:"text"`

	// type=thinking
	Thinking  string `json:"thinking"`
	Signature string `json:"signature,omitempty"`

	// type=image
	Source *AnthropicImageSource `json:"source,omitempty"`

	// type=tool_use / server_tool_use / mcp_tool_use
	ID         string          `json:"id,omitempty"`
	Name       string          `json:"name,omitempty"`
	ServerName string          `json:"server_name,omitempty"`
	Input      json.RawMessage `json:"input,omitempty"`

	// type=tool_result / web_search_tool_result / mcp_tool_result
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"` // string or []AnthropicContentBlock
	IsError   bool            `json:"is_error,omitempty"`
}

// AnthropicImageSource describes the source data for an image content block.
type AnthropicImageSource struct {
	Type      string `json:"type"` // "base64"
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

// AnthropicTool describes a tool available to the model.
type AnthropicTool struct {
	Type          string          `json:"type,omitempty"` // e.g. "web_search_20250305" / "mcp_toolset"
	Name          string          `json:"name,omitempty"`
	Description   string          `json:"description,omitempty"`
	InputSchema   json.RawMessage `json:"input_schema,omitempty"`
	MCPServerName string          `json:"mcp_server_name,omitempty"`
	DefaultConfig json.RawMessage `json:"default_config,omitempty"`
	Configs       json.RawMessage `json:"configs,omitempty"`
	CacheControl  json.RawMessage `json:"cache_control,omitempty"`
}

// AnthropicMCPServer describes one entry in the mcp_servers array.
type AnthropicMCPServer struct {
	Type               string          `json:"type,omitempty"`
	URL                string          `json:"url,omitempty"`
	Name               string          `json:"name,omitempty"`
	AuthorizationToken string          `json:"authorization_token,omitempty"`
	ToolConfiguration  json.RawMessage `json:"tool_configuration,omitempty"`
}

// AnthropicResponse is the non-streaming response from POST /v1/messages.
type AnthropicResponse struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"` // "message"
	Role         string                  `json:"role"` // "assistant"
	Content      []AnthropicContentBlock `json:"content"`
	Model        string                  `json:"model"`
	StopReason   string                  `json:"stop_reason"`
	StopSequence *string                 `json:"stop_sequence,omitempty"`
	Usage        AnthropicUsage          `json:"usage"`
	Container    json.RawMessage         `json:"container,omitempty"`
}

// AnthropicUsage holds token counts in Anthropic format.
type AnthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

// ---------------------------------------------------------------------------
// Anthropic SSE event types
// ---------------------------------------------------------------------------

// AnthropicStreamEvent is a single SSE event in the Anthropic streaming protocol.
type AnthropicStreamEvent struct {
	Type string `json:"type"`

	// message_start
	Message *AnthropicResponse `json:"message,omitempty"`

	// content_block_start
	Index        *int                   `json:"index,omitempty"`
	ContentBlock *AnthropicContentBlock `json:"content_block,omitempty"`

	// content_block_delta
	Delta *AnthropicDelta `json:"delta,omitempty"`

	// message_delta
	Usage *AnthropicUsage `json:"usage,omitempty"`

	// error
	Error *AnthropicErrorDetail `json:"error,omitempty"`
}

// AnthropicErrorDetail describes an error in a streaming error event.
type AnthropicErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// AnthropicDelta carries incremental content in streaming events.
type AnthropicDelta struct {
	Type string `json:"type,omitempty"` // "text_delta" | "input_json_delta" | "thinking_delta" | "signature_delta"

	// text_delta
	Text string `json:"text,omitempty"`

	// input_json_delta
	PartialJSON string `json:"partial_json,omitempty"`

	// thinking_delta
	Thinking string `json:"thinking,omitempty"`

	// signature_delta
	Signature string `json:"signature,omitempty"`

	// message_delta fields
	StopReason   string  `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
}

// ---------------------------------------------------------------------------
// OpenAI Responses API types
// ---------------------------------------------------------------------------

// ResponsesRequest is the request body for POST /v1/responses.
type ResponsesRequest struct {
	Background           *bool               `json:"background,omitempty"`
	ContextManagement    json.RawMessage     `json:"context_management,omitempty"`
	Conversation         json.RawMessage     `json:"conversation,omitempty"`
	Include              []string            `json:"include,omitempty"`
	Input                json.RawMessage     `json:"input"` // string or []ResponsesInputItem
	Instructions         string              `json:"instructions,omitempty"`
	MaxOutputTokens      *int                `json:"max_output_tokens,omitempty"`
	MaxToolCalls         *int                `json:"max_tool_calls,omitempty"`
	Metadata             json.RawMessage     `json:"metadata,omitempty"`
	Model                string              `json:"model"`
	ParallelToolCalls    *bool               `json:"parallel_tool_calls,omitempty"`
	PreviousResponseID   string              `json:"previous_response_id,omitempty"`
	Prompt               *ResponsesPrompt    `json:"prompt,omitempty"`
	PromptCacheKey       string              `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention string              `json:"prompt_cache_retention,omitempty"`
	Reasoning            *ResponsesReasoning `json:"reasoning,omitempty"`
	SafetyIdentifier     string              `json:"safety_identifier,omitempty"`
	ServiceTier          string              `json:"service_tier,omitempty"`
	Store                *bool               `json:"store,omitempty"`
	Stream               bool                `json:"stream,omitempty"`
	StreamOptions        json.RawMessage     `json:"stream_options,omitempty"`
	Temperature          *float64            `json:"temperature,omitempty"`
	Text                 json.RawMessage     `json:"text,omitempty"`
	ToolChoice           json.RawMessage     `json:"tool_choice,omitempty"`
	Tools                []ResponsesTool     `json:"tools,omitempty"`
	TopLogprobs          *int                `json:"top_logprobs,omitempty"`
	TopP                 *float64            `json:"top_p,omitempty"`
	Truncation           string              `json:"truncation,omitempty"`
	User                 string              `json:"user,omitempty"`
}

// ResponsesPrompt references a reusable prompt template.
type ResponsesPrompt struct {
	ID        string         `json:"id"`
	Variables map[string]any `json:"variables,omitempty"`
	Version   string         `json:"version,omitempty"`
}

// ResponsesReasoning configures reasoning effort in the Responses API.
type ResponsesReasoning struct {
	Effort           string `json:"effort,omitempty"`  // "low" | "medium" | "high" | "xhigh"
	Summary          string `json:"summary,omitempty"` // "auto" | "concise" | "detailed"
	GenerateSummary  string `json:"generate_summary,omitempty"`
	EncryptedContent string `json:"encrypted_content,omitempty"`
}

// ResponsesInputItem is one item in the Responses API input array.
// The Type field determines which other fields are populated.
type ResponsesInputItem struct {
	// Common
	Type string `json:"type,omitempty"` // "" for role-based messages

	// Role-based messages (system/user/assistant)
	Role    string          `json:"role,omitempty"`
	Content json.RawMessage `json:"content,omitempty"` // string or []ResponsesContentPart
	Phase   string          `json:"phase,omitempty"`

	// type=function_call / function_call_output / tool-like items
	CallID    string          `json:"call_id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments string          `json:"arguments,omitempty"`
	ID        string          `json:"id,omitempty"`
	Status    string          `json:"status,omitempty"` // "in_progress" | "completed" | "incomplete"
	Output    string          `json:"output,omitempty"`
	OutputRaw json.RawMessage `json:"output_raw,omitempty"`
	Namespace string          `json:"namespace,omitempty"`
	RawItem   json.RawMessage `json:"item,omitempty"`

	// type=reasoning / compaction
	Summary          []ResponsesSummary `json:"summary,omitempty"`
	EncryptedContent string             `json:"encrypted_content,omitempty"`
}

// ResponsesContentPart is a typed content part in a Responses message.
type ResponsesContentPart struct {
	Type        string             `json:"type"`
	Text        string             `json:"text,omitempty"`
	ImageURL    string             `json:"image_url,omitempty"`
	FileID      string             `json:"file_id,omitempty"`
	FileURL     string             `json:"file_url,omitempty"`
	FileData    string             `json:"file_data,omitempty"`
	Filename    string             `json:"filename,omitempty"`
	Detail      string             `json:"detail,omitempty"`
	InputAudio  *ResponsesAudioRef `json:"input_audio,omitempty"`
	Refusal     string             `json:"refusal,omitempty"`
	Annotations []json.RawMessage  `json:"annotations,omitempty"`
	Logprobs    json.RawMessage    `json:"logprobs,omitempty"`
}

// ResponsesAudioRef contains inline audio input metadata.
type ResponsesAudioRef struct {
	Data   string `json:"data,omitempty"`
	Format string `json:"format,omitempty"`
}

// ResponsesTool describes a tool in the Responses API.
type ResponsesTool struct {
	Type            string          `json:"type"`
	Name            string          `json:"name,omitempty"`
	Description     string          `json:"description,omitempty"`
	Parameters      json.RawMessage `json:"parameters,omitempty"`
	Strict          *bool           `json:"strict,omitempty"`
	ServerLabel     string          `json:"server_label,omitempty"`
	ConnectorID     string          `json:"connector_id,omitempty"`
	ServerURL       string          `json:"server_url,omitempty"`
	Authorization   string          `json:"authorization,omitempty"`
	AllowedTools    json.RawMessage `json:"allowed_tools,omitempty"`
	Headers         json.RawMessage `json:"headers,omitempty"`
	RequireApproval string          `json:"require_approval,omitempty"`
}

// ResponsesResponse is the non-streaming response from POST /v1/responses.
type ResponsesResponse struct {
	ID                string                      `json:"id"`
	Object            string                      `json:"object"` // "response"
	CreatedAt         int64                       `json:"created_at,omitempty"`
	Error             *ResponsesError             `json:"error,omitempty"`
	IncompleteDetails *ResponsesIncompleteDetails `json:"incomplete_details,omitempty"`
	Instructions      string                      `json:"instructions,omitempty"`
	Metadata          json.RawMessage             `json:"metadata,omitempty"`
	Model             string                      `json:"model"`
	Output            []ResponsesOutput           `json:"output"`
	OutputText        string                      `json:"output_text,omitempty"`
	ParallelToolCalls *bool                       `json:"parallel_tool_calls,omitempty"`
	Status            string                      `json:"status"` // "completed" | "incomplete" | "failed"
	Temperature       *float64                    `json:"temperature,omitempty"`
	Text              json.RawMessage             `json:"text,omitempty"`
	ToolChoice        json.RawMessage             `json:"tool_choice,omitempty"`
	Tools             []ResponsesTool             `json:"tools,omitempty"`
	TopP              *float64                    `json:"top_p,omitempty"`
	Truncation        string                      `json:"truncation,omitempty"`
	Usage             *ResponsesUsage             `json:"usage,omitempty"`
	User              string                      `json:"user,omitempty"`
	Conversation      *ResponsesConversation      `json:"conversation,omitempty"`
}

// ResponsesConversation identifies a stored conversation.
type ResponsesConversation struct {
	ID string `json:"id,omitempty"`
}

// ResponsesError describes an error in a failed response.
type ResponsesError struct {
	Code    string `json:"code,omitempty"`
	Message string `json:"message"`
}

// ResponsesIncompleteDetails explains why a response is incomplete.
type ResponsesIncompleteDetails struct {
	Reason string `json:"reason,omitempty"` // "max_output_tokens" | "content_filter"
}

// ResponsesOutput is one output item in a Responses API response.
type ResponsesOutput struct {
	Type string `json:"type"`

	// Common item identity
	ID        string `json:"id,omitempty"`
	Status    string `json:"status,omitempty"`
	CreatedBy string `json:"created_by,omitempty"`

	// type=message
	Role    string                 `json:"role,omitempty"`
	Content []ResponsesContentPart `json:"content,omitempty"`
	Phase   string                 `json:"phase,omitempty"`

	// type=reasoning / compaction
	EncryptedContent string             `json:"encrypted_content,omitempty"`
	Summary          []ResponsesSummary `json:"summary,omitempty"`

	// type=function_call / tool-like items
	CallID            string           `json:"call_id,omitempty"`
	Name              string           `json:"name,omitempty"`
	Arguments         string           `json:"arguments,omitempty"`
	Namespace         string           `json:"namespace,omitempty"`
	Action            *WebSearchAction `json:"action,omitempty"`
	Actions           json.RawMessage  `json:"actions,omitempty"`
	Result            string           `json:"result,omitempty"`
	Output            json.RawMessage  `json:"output,omitempty"`
	Logs              string           `json:"logs,omitempty"`
	Code              string           `json:"code,omitempty"`
	Results           json.RawMessage  `json:"results,omitempty"`
	ServerLabel       string           `json:"server_label,omitempty"`
	ApprovalRequestID string           `json:"approval_request_id,omitempty"`
	Approved          *bool            `json:"approved,omitempty"`
	Reason            string           `json:"reason,omitempty"`
	Operation         json.RawMessage  `json:"operation,omitempty"`
	RawItem           json.RawMessage  `json:"item,omitempty"`
}

// ResponsesOutputPart captures generic content-part events.
type ResponsesOutputPart struct {
	Type        string            `json:"type"`
	Text        string            `json:"text,omitempty"`
	Refusal     string            `json:"refusal,omitempty"`
	Annotations []json.RawMessage `json:"annotations,omitempty"`
	RawPart     json.RawMessage   `json:"part,omitempty"`
}

// ResponsesAnnotationAdded captures output text annotation events.
type ResponsesAnnotationAdded struct {
	Annotation json.RawMessage `json:"annotation,omitempty"`
}

// WebSearchAction describes the search action in a web_search_call output item.
type WebSearchAction struct {
	Type  string `json:"type,omitempty"`  // "search"
	Query string `json:"query,omitempty"` // primary search query
}

// ResponsesSummary is a summary text block inside a reasoning output.
type ResponsesSummary struct {
	Type string `json:"type"` // "summary_text"
	Text string `json:"text"`
}

// ResponsesUsage holds token counts in Responses API format.
type ResponsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`

	// Optional detailed breakdown
	InputTokensDetails  *ResponsesInputTokensDetails  `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *ResponsesOutputTokensDetails `json:"output_tokens_details,omitempty"`
}

// ResponsesInputTokensDetails breaks down input token usage.
type ResponsesInputTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// ResponsesOutputTokensDetails breaks down output token usage.
type ResponsesOutputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// ---------------------------------------------------------------------------
// Responses SSE event types
// ---------------------------------------------------------------------------

// ResponsesStreamEvent is a single SSE event in the Responses streaming protocol.
// The Type field corresponds to the "type" in the JSON payload.
type ResponsesStreamEvent struct {
	Type string `json:"type"`

	// response.* lifecycle events
	Response *ResponsesResponse `json:"response,omitempty"`

	// response.output_item.added / response.output_item.done
	Item *ResponsesOutput `json:"item,omitempty"`

	// Generic output/content addressing
	OutputIndex     int `json:"output_index,omitempty"`
	ContentIndex    int `json:"content_index,omitempty"`
	SummaryIndex    int `json:"summary_index,omitempty"`
	AnnotationIndex int `json:"annotation_index,omitempty"`

	// Common delta/done payloads
	Delta       string          `json:"delta,omitempty"`
	Text        string          `json:"text,omitempty"`
	Refusal     string          `json:"refusal,omitempty"`
	ItemID      string          `json:"item_id,omitempty"`
	CallID      string          `json:"call_id,omitempty"`
	Name        string          `json:"name,omitempty"`
	Arguments   string          `json:"arguments,omitempty"`
	Part        json.RawMessage `json:"part,omitempty"`
	Annotation  json.RawMessage `json:"annotation,omitempty"`
	Logprobs    json.RawMessage `json:"logprobs,omitempty"`
	Obfuscation string          `json:"obfuscation,omitempty"`

	// error event fields
	Code    string `json:"code,omitempty"`
	Param   string `json:"param,omitempty"`
	Message string `json:"message,omitempty"`

	// Sequence number for ordering events
	SequenceNumber int `json:"sequence_number,omitempty"`
}

// ---------------------------------------------------------------------------
// OpenAI Chat Completions API types
// ---------------------------------------------------------------------------

// ChatCompletionsRequest is the request body for POST /v1/chat/completions.
type ChatCompletionsRequest struct {
	Model               string             `json:"model"`
	Messages            []ChatMessage      `json:"messages"`
	Instructions        string             `json:"instructions,omitempty"` // OpenAI Responses API compat
	MaxTokens           *int               `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int               `json:"max_completion_tokens,omitempty"`
	Temperature         *float64           `json:"temperature,omitempty"`
	TopP                *float64           `json:"top_p,omitempty"`
	Stream              bool               `json:"stream,omitempty"`
	StreamOptions       *ChatStreamOptions `json:"stream_options,omitempty"`
	Tools               []ChatTool         `json:"tools,omitempty"`
	ToolChoice          json.RawMessage    `json:"tool_choice,omitempty"`
	ReasoningEffort     string             `json:"reasoning_effort,omitempty"` // "low" | "medium" | "high" | "xhigh"
	ServiceTier         string             `json:"service_tier,omitempty"`
	Stop                json.RawMessage    `json:"stop,omitempty"` // string or []string

	// Legacy function calling (deprecated but still supported)
	Functions    []ChatFunction  `json:"functions,omitempty"`
	FunctionCall json.RawMessage `json:"function_call,omitempty"`
}

// ChatStreamOptions configures streaming behavior.
type ChatStreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// ChatMessage is a single message in the Chat Completions conversation.
type ChatMessage struct {
	Role             string          `json:"role"` // "system" | "user" | "assistant" | "tool" | "function"
	Content          json.RawMessage `json:"content,omitempty"`
	ReasoningContent string          `json:"reasoning_content,omitempty"`
	Name             string          `json:"name,omitempty"`
	ToolCalls        []ChatToolCall  `json:"tool_calls,omitempty"`
	ToolCallID       string          `json:"tool_call_id,omitempty"`

	// Legacy function calling
	FunctionCall *ChatFunctionCall `json:"function_call,omitempty"`
}

// ChatContentPart is a typed content part in a multi-modal message.
type ChatContentPart struct {
	Type     string        `json:"type"` // "text" | "image_url"
	Text     string        `json:"text,omitempty"`
	ImageURL *ChatImageURL `json:"image_url,omitempty"`
}

// ChatImageURL contains the URL for an image content part.
type ChatImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"` // "auto" | "low" | "high"
}

// ChatTool describes a tool available to the model.
type ChatTool struct {
	Type     string        `json:"type"` // "function"
	Function *ChatFunction `json:"function,omitempty"`
}

// ChatFunction describes a function tool definition.
type ChatFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

// ChatToolCall represents a tool call made by the assistant.
// Index is only populated in streaming chunks (omitted in non-streaming responses).
type ChatToolCall struct {
	Index    *int             `json:"index,omitempty"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"` // "function"
	Function ChatFunctionCall `json:"function"`
}

// ChatFunctionCall contains the function name and arguments.
type ChatFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ChatCompletionsResponse is the non-streaming response from POST /v1/chat/completions.
type ChatCompletionsResponse struct {
	ID                string       `json:"id"`
	Object            string       `json:"object"` // "chat.completion"
	Created           int64        `json:"created"`
	Model             string       `json:"model"`
	Choices           []ChatChoice `json:"choices"`
	Usage             *ChatUsage   `json:"usage,omitempty"`
	SystemFingerprint string       `json:"system_fingerprint,omitempty"`
	ServiceTier       string       `json:"service_tier,omitempty"`
}

// ChatChoice is a single completion choice.
type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"` // "stop" | "length" | "tool_calls" | "content_filter"
}

// ChatUsage holds token counts in Chat Completions format.
type ChatUsage struct {
	PromptTokens        int               `json:"prompt_tokens"`
	CompletionTokens    int               `json:"completion_tokens"`
	TotalTokens         int               `json:"total_tokens"`
	PromptTokensDetails *ChatTokenDetails `json:"prompt_tokens_details,omitempty"`
}

// ChatTokenDetails provides a breakdown of token usage.
type ChatTokenDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// ChatCompletionsChunk is a single streaming chunk from POST /v1/chat/completions.
type ChatCompletionsChunk struct {
	ID                string            `json:"id"`
	Object            string            `json:"object"` // "chat.completion.chunk"
	Created           int64             `json:"created"`
	Model             string            `json:"model"`
	Choices           []ChatChunkChoice `json:"choices"`
	Usage             *ChatUsage        `json:"usage,omitempty"`
	SystemFingerprint string            `json:"system_fingerprint,omitempty"`
	ServiceTier       string            `json:"service_tier,omitempty"`
}

// ChatChunkChoice is a single choice in a streaming chunk.
type ChatChunkChoice struct {
	Index        int       `json:"index"`
	Delta        ChatDelta `json:"delta"`
	FinishReason *string   `json:"finish_reason"` // pointer: null when not final
}

// ChatDelta carries incremental content in a streaming chunk.
type ChatDelta struct {
	Role             string         `json:"role,omitempty"`
	Content          *string        `json:"content,omitempty"` // pointer: omit when not present, null vs "" matters
	ReasoningContent *string        `json:"reasoning_content,omitempty"`
	ToolCalls        []ChatToolCall `json:"tool_calls,omitempty"`
}

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

// minMaxOutputTokens is the floor for max_output_tokens in a Responses request.
// Very small values may cause upstream API errors, so we enforce a minimum.
const minMaxOutputTokens = 128

// minAnthropicMaxOutputTokens is the floor for max_output_tokens when converting
// Anthropic requests to Responses API. Claude Code expects a higher minimum.
const minAnthropicMaxOutputTokens = 12800
