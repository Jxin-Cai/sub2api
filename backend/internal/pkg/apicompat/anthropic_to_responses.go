package apicompat

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

// AnthropicToResponses converts an Anthropic Messages request directly into
// a Responses API request. This preserves fields that would be lost in a
// Chat Completions intermediary round-trip (e.g. thinking, cache_control,
// structured system prompts).
func AnthropicToResponses(req *AnthropicRequest) (*ResponsesRequest, error) {
	input, err := convertAnthropicToResponsesInput(nil, req.Messages)
	if err != nil {
		return nil, err
	}

	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}

	// System prompt → instructions field (not input item)
	instructions := ""
	if len(req.System) > 0 {
		instructions, err = parseAnthropicSystemPrompt(req.System)
		if err != nil {
			return nil, err
		}
	}

	tempOne := 1.0
	out := &ResponsesRequest{
		Model:        req.Model,
		Instructions: instructions,
		Input:        inputJSON,
		Temperature:  &tempOne,
		TopP:         req.TopP,
		Stream:       req.Stream,
		Include:      []string{"reasoning.encrypted_content"},
	}

	storeFalse := false
	out.Store = &storeFalse

	parallelTrue := true
	out.ParallelToolCalls = &parallelTrue

	if req.MaxTokens > 0 {
		v := req.MaxTokens
		if v < minAnthropicMaxOutputTokens {
			v = minAnthropicMaxOutputTokens
		}
		out.MaxOutputTokens = &v
	}

	if len(req.Metadata) > 0 {
		out.Metadata = req.Metadata
	}

	if req.ServiceTier != "" {
		out.ServiceTier = req.ServiceTier
	}

	if len(req.ContextManagement) > 0 {
		out.ContextManagement = anthropicContextManagementToResponses(req.ContextManagement)
	}

	if len(req.Container) > 0 {
		out.Conversation = req.Container
	}

	if len(req.Tools) > 0 {
		out.Tools = convertAnthropicToolsToResponses(req.Tools, req.MCPServers)
	} else if len(req.MCPServers) > 0 {
		out.Tools = convertAnthropicToolsToResponses(nil, req.MCPServers)
	}

	if req.OutputConfig != nil && len(req.OutputConfig.Format) > 0 {
		out.Text = json.RawMessage(`{"format":` + string(req.OutputConfig.Format) + `}`)
	}

	effort := "high"
	if req.OutputConfig != nil && req.OutputConfig.Effort != "" {
		effort = req.OutputConfig.Effort
	}
	normalizedEffort := mapAnthropicEffortToResponses(effort)
	out.Reasoning = &ResponsesReasoning{
		Effort:  normalizedEffort,
		Summary: "detailed",
	}

	if len(req.ToolChoice) > 0 {
		tc, parallelToolCalls, err := convertAnthropicToolChoiceToResponses(req.ToolChoice)
		if err != nil {
			return nil, fmt.Errorf("convert tool_choice: %w", err)
		}
		out.ToolChoice = tc
		if parallelToolCalls != nil {
			out.ParallelToolCalls = parallelToolCalls
		}
	}

	return out, nil
}

// convertAnthropicToolChoiceToResponses maps Anthropic tool_choice to Responses format.
//
//	{"type":"auto"}            → "auto"
//	{"type":"any"}             → "required"
//	{"type":"none"}            → "none"
//	{"type":"tool","name":"X"} → {"type":"function","function":{"name":"X"}}
func convertAnthropicToolChoiceToResponses(raw json.RawMessage) (json.RawMessage, *bool, error) {
	var tc struct {
		Type                   string `json:"type"`
		Name                   string `json:"name"`
		DisableParallelToolUse *bool  `json:"disable_parallel_tool_use,omitempty"`
	}
	if err := json.Unmarshal(raw, &tc); err != nil {
		return nil, nil, err
	}

	var parallelToolCalls *bool
	if tc.DisableParallelToolUse != nil {
		value := !*tc.DisableParallelToolUse
		parallelToolCalls = &value
	}

	switch tc.Type {
	case "auto":
		encoded, err := json.Marshal("auto")
		return encoded, parallelToolCalls, err
	case "any":
		encoded, err := json.Marshal("required")
		return encoded, parallelToolCalls, err
	case "none":
		encoded, err := json.Marshal("none")
		return encoded, parallelToolCalls, err
	case "tool":
		encoded, err := json.Marshal(map[string]any{
			"type":     "function",
			"function": map[string]string{"name": tc.Name},
		})
		return encoded, parallelToolCalls, err
	default:
		return nil, parallelToolCalls, nil
	}
}

// convertAnthropicToResponsesInput builds the Responses API input items array
// from the Anthropic system field and message list.
func convertAnthropicToResponsesInput(system json.RawMessage, msgs []AnthropicMessage) ([]ResponsesInputItem, error) {
	var out []ResponsesInputItem

	// System prompt → system role input item.
	if len(system) > 0 {
		sysText, err := parseAnthropicSystemPrompt(system)
		if err != nil {
			return nil, err
		}
		if sysText != "" {
			content, _ := json.Marshal(sysText)
			out = append(out, ResponsesInputItem{
				Role:    "system",
				Content: content,
			})
		}
	}

	for _, m := range msgs {
		items, err := anthropicMsgToResponsesItems(m)
		if err != nil {
			return nil, err
		}
		out = append(out, items...)
	}
	return out, nil
}

func anthropicContextManagementToResponses(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	var edits []json.RawMessage
	if err := json.Unmarshal(raw, &edits); err == nil {
		return normalizeAnthropicContextManagementEditsToResponses(edits)
	}
	var payload struct {
		Edits []json.RawMessage `json:"edits,omitempty"`
	}
	if err := json.Unmarshal(raw, &payload); err == nil {
		if len(payload.Edits) == 0 {
			return nil
		}
		return normalizeAnthropicContextManagementEditsToResponses(payload.Edits)
	}
	return nil
}

func normalizeAnthropicContextManagementEditsToResponses(edits []json.RawMessage) json.RawMessage {
	if len(edits) == 0 {
		return nil
	}
	converted := make([]json.RawMessage, 0, len(edits))
	for _, edit := range edits {
		converted = append(converted, anthropicContextManagementEditToResponses(edit))
	}
	encoded, err := json.Marshal(converted)
	if err != nil {
		return nil
	}
	return encoded
}

func anthropicContextManagementEditToResponses(raw json.RawMessage) json.RawMessage {
	var edit map[string]any
	if err := json.Unmarshal(raw, &edit); err != nil {
		return raw
	}
	if strings.TrimSpace(stringValue(edit["type"])) != "compact_20260112" {
		return raw
	}
	trigger, _ := edit["trigger"].(map[string]any)
	value, ok := intValue(trigger["value"])
	if !ok {
		return mustMarshalJSON(map[string]any{"type": "compaction"})
	}
	return mustMarshalJSON(map[string]any{"type": "compaction", "compact_threshold": value})
}

// parseAnthropicSystemPrompt handles the Anthropic system field which can be
// a plain string or an array of text blocks.
func parseAnthropicSystemPrompt(raw json.RawMessage) (string, error) {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}
	var blocks []AnthropicContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", err
	}
	var parts []string
	for _, b := range blocks {
		if b.Type == "text" && b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n\n"), nil
}

// anthropicMsgToResponsesItems converts a single Anthropic message into one
// or more Responses API input items.
func anthropicMsgToResponsesItems(m AnthropicMessage) ([]ResponsesInputItem, error) {
	switch m.Role {
	case "user":
		return anthropicUserToResponses(m.Content)
	case "assistant":
		return anthropicAssistantToResponses(m.Content)
	default:
		return anthropicUserToResponses(m.Content)
	}
}

// anthropicUserToResponses handles an Anthropic user message. Content can be a
// plain string or an array of blocks. tool_result blocks are extracted into
// function_call_output items. Image blocks are converted to input_image parts.
func anthropicUserToResponses(raw json.RawMessage) ([]ResponsesInputItem, error) {
	// Try plain string.
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		content, _ := json.Marshal(s)
		return []ResponsesInputItem{{Role: "user", Content: content}}, nil
	}

	var blocks []AnthropicContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, err
	}

	var out []ResponsesInputItem
	var toolResultImageParts []ResponsesContentPart

	// Extract tool_result blocks → function_call_output items.
	// Images inside tool_results are extracted separately because the
	// Responses API function_call_output.output only accepts strings.
	for _, b := range blocks {
		if !isAnthropicToolResultBlockType(b.Type) {
			continue
		}
		outputText, imageParts := convertToolResultOutput(b)
		status := "completed"
		if b.IsError {
			status = "incomplete"
		}
		item := ResponsesInputItem{
			Type:      "function_call_output",
			CallID:    b.ToolUseID,
			Output:    outputText,
			OutputRaw: b.Content,
			Status:    status,
		}
		if b.Type == "mcp_tool_result" {
			item.Namespace = "mcp"
			item.RawItem = mustMarshalAnthropicContentBlock(b)
		}
		out = append(out, item)
		toolResultImageParts = append(toolResultImageParts, imageParts...)
	}

	// Remaining text + image blocks → user message with content parts.
	// Also include images extracted from tool_results so the model can see them.
	var parts []ResponsesContentPart
	for _, b := range blocks {
		switch b.Type {
		case "text":
			if b.Text != "" {
				parts = append(parts, ResponsesContentPart{Type: "input_text", Text: b.Text})
			}
		case "image":
			if uri := anthropicImageToDataURI(b.Source); uri != "" {
				parts = append(parts, ResponsesContentPart{Type: "input_image", ImageURL: uri})
			}
		}
	}
	parts = append(parts, toolResultImageParts...)

	if len(parts) > 0 {
		content, err := json.Marshal(parts)
		if err != nil {
			return nil, err
		}
		out = append(out, ResponsesInputItem{Role: "user", Content: content})
	}

	return out, nil
}

// anthropicAssistantToResponses handles an Anthropic assistant message.
// Text content → assistant message with output_text parts.
// tool_use blocks → function_call items.
// thinking blocks with signature → reasoning or compaction items.
func anthropicAssistantToResponses(raw json.RawMessage) ([]ResponsesInputItem, error) {
	// Try plain string.
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		parts := []ResponsesContentPart{{Type: "output_text", Text: s}}
		partsJSON, err := json.Marshal(parts)
		if err != nil {
			return nil, err
		}
		return []ResponsesInputItem{{Role: "assistant", Content: partsJSON}}, nil
	}

	var blocks []AnthropicContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, err
	}

	var items []ResponsesInputItem
	var textParts []ResponsesContentPart

	for _, b := range blocks {
		switch b.Type {
		case "text":
			if b.Text != "" {
				textParts = append(textParts, ResponsesContentPart{Type: "output_text", Text: b.Text})
			}
		case "tool_use", "mcp_tool_use":
			// Flush pending text before tool_use
			if len(textParts) > 0 {
				partsJSON, err := json.Marshal(textParts)
				if err != nil {
					return nil, err
				}
				items = append(items, ResponsesInputItem{Role: "assistant", Content: partsJSON})
				textParts = nil
			}
			args := "{}"
			if len(b.Input) > 0 {
				args = string(b.Input)
			}
			item := ResponsesInputItem{
				Type:      "function_call",
				CallID:    b.ID,
				Name:      b.Name,
				Arguments: args,
				Status:    "completed",
			}
			if b.Type == "mcp_tool_use" {
				item.Namespace = b.ServerName
				if item.Namespace == "" {
					item.Namespace = "mcp"
				}
				item.RawItem = mustMarshalAnthropicContentBlock(b)
			}
			items = append(items, item)
		case "thinking":
			if b.Signature == "" {
				continue
			}
			// Flush pending text before reasoning/compaction
			if len(textParts) > 0 {
				partsJSON, err := json.Marshal(textParts)
				if err != nil {
					return nil, err
				}
				items = append(items, ResponsesInputItem{Role: "assistant", Content: partsJSON})
				textParts = nil
			}
			if envelope := decodeOpenAIReasoningSignatureEnvelope(b.Signature); envelope != nil {
				item := ResponsesInputItem{
					Type:             envelope.Type,
					ID:               envelope.ID,
					EncryptedContent: envelope.EncryptedContent,
					Summary:          envelope.Summary,
					Status:           envelope.Status,
				}
				if item.Type == "reasoning" && len(item.Summary) == 0 {
					item.Summary = summaryFromThinkingBlock(b.Thinking)
				}
				items = append(items, item)
				continue
			}
			if compaction := decodeCompactionSignature(b.Signature); compaction != nil {
				items = append(items, ResponsesInputItem{
					Type:             "compaction",
					ID:               compaction.id,
					EncryptedContent: compaction.encryptedContent,
				})
				continue
			}
			if strings.Contains(b.Signature, "@") {
				enc, id := parseReasoningSignature(b.Signature)
				if len(id) > maxReasoningIDLength {
					continue
				}
				items = append(items, ResponsesInputItem{
					Type:             "reasoning",
					ID:               id,
					EncryptedContent: enc,
					Summary:          summaryFromThinkingBlock(b.Thinking),
				})
			}
		}
	}

	// Flush remaining text
	if len(textParts) > 0 {
		partsJSON, err := json.Marshal(textParts)
		if err != nil {
			return nil, err
		}
		items = append(items, ResponsesInputItem{Role: "assistant", Content: partsJSON})
	}

	return items, nil
}

func isAnthropicToolResultBlockType(blockType string) bool {
	return blockType == "tool_result" || blockType == "mcp_tool_result"
}

func mustMarshalAnthropicContentBlock(block AnthropicContentBlock) json.RawMessage {
	payload, err := json.Marshal(block)
	if err != nil {
		return nil
	}
	return payload
}

const (
	thinkingPlaceholder             = "Thinking..."
	openAIReasoningSignaturePrefix  = "oai2#"
	openAIReasoningSignatureVersion = 2
	compactionSignaturePrefix       = "cm1#"
	compactionSignatureSeparator    = "@"
	maxReasoningIDLength            = 64
)

type openAIReasoningSignatureEnvelope struct {
	Version          int                `json:"v"`
	Type             string             `json:"type"`
	ID               string             `json:"id,omitempty"`
	EncryptedContent string             `json:"encrypted_content,omitempty"`
	Summary          []ResponsesSummary `json:"summary,omitempty"`
	Status           string             `json:"status,omitempty"`
	Model            string             `json:"model,omitempty"`
}

type compactionCarrier struct {
	id               string
	encryptedContent string
}

func encodeReasoningItemSignature(item ResponsesOutput) string {
	return encodeOpenAIReasoningSignatureEnvelope(openAIReasoningSignatureEnvelope{
		Version:          openAIReasoningSignatureVersion,
		Type:             "reasoning",
		ID:               item.ID,
		EncryptedContent: item.EncryptedContent,
		Summary:          item.Summary,
		Status:           item.Status,
	})
}

func encodeCompactionItemSignature(item ResponsesOutput) string {
	return encodeOpenAIReasoningSignatureEnvelope(openAIReasoningSignatureEnvelope{
		Version:          openAIReasoningSignatureVersion,
		Type:             "compaction",
		ID:               item.ID,
		EncryptedContent: item.EncryptedContent,
		Summary:          item.Summary,
		Status:           item.Status,
	})
}

func encodeOpenAIReasoningSignatureEnvelope(envelope openAIReasoningSignatureEnvelope) string {
	encoded, err := json.Marshal(envelope)
	if err != nil {
		return ""
	}
	return openAIReasoningSignaturePrefix + base64.RawURLEncoding.EncodeToString(encoded)
}

func decodeOpenAIReasoningSignatureEnvelope(sig string) *openAIReasoningSignatureEnvelope {
	if !strings.HasPrefix(sig, openAIReasoningSignaturePrefix) {
		return nil
	}
	payload := strings.TrimPrefix(sig, openAIReasoningSignaturePrefix)
	decoded, err := base64.RawURLEncoding.DecodeString(payload)
	if err != nil {
		return nil
	}
	var envelope openAIReasoningSignatureEnvelope
	if err := json.Unmarshal(decoded, &envelope); err != nil {
		return nil
	}
	if envelope.Version != openAIReasoningSignatureVersion {
		return nil
	}
	switch envelope.Type {
	case "reasoning", "compaction":
		return &envelope
	default:
		return nil
	}
}

func summaryFromThinkingBlock(thinking string) []ResponsesSummary {
	if thinking == thinkingPlaceholder {
		thinking = ""
	}
	return []ResponsesSummary{{Type: "summary_text", Text: thinking}}
}

// encodeCompactionSignature encodes a compaction carrier into a signature string.
func encodeCompactionSignature(id, encryptedContent string) string {
	return compactionSignaturePrefix + encryptedContent + compactionSignatureSeparator + id
}

// decodeCompactionSignature decodes a cm1#-prefixed signature into its parts.
func decodeCompactionSignature(sig string) *compactionCarrier {
	if !strings.HasPrefix(sig, compactionSignaturePrefix) {
		return nil
	}
	raw := sig[len(compactionSignaturePrefix):]
	sepIdx := strings.LastIndex(raw, compactionSignatureSeparator)
	if sepIdx <= 0 || sepIdx == len(raw)-1 {
		return nil
	}
	enc := raw[:sepIdx]
	id := raw[sepIdx+1:]
	if enc == "" {
		return nil
	}
	return &compactionCarrier{id: id, encryptedContent: enc}
}

// parseReasoningSignature splits "encrypted_content@id" into its parts.
func parseReasoningSignature(sig string) (encryptedContent, id string) {
	idx := strings.LastIndex(sig, "@")
	if idx <= 0 || idx == len(sig)-1 {
		return sig, ""
	}
	return sig[:idx], sig[idx+1:]
}

// anthropicImageToDataURI converts an AnthropicImageSource to a data URI string.
// Returns "" if the source is nil or has no data.
func anthropicImageToDataURI(src *AnthropicImageSource) string {
	if src == nil || src.Data == "" {
		return ""
	}
	mediaType := src.MediaType
	if mediaType == "" {
		mediaType = "image/png"
	}
	return "data:" + mediaType + ";base64," + src.Data
}

// convertToolResultOutput extracts text and image content from a tool_result
// block. Returns the text as a string for the function_call_output Output
// field, plus any image parts that must be sent in a separate user message
// (the Responses API output field only accepts strings).
func convertToolResultOutput(b AnthropicContentBlock) (string, []ResponsesContentPart) {
	if len(b.Content) == 0 {
		return "(empty)", nil
	}

	// Try plain string content.
	var s string
	if err := json.Unmarshal(b.Content, &s); err == nil {
		if s == "" {
			s = "(empty)"
		}
		return s, nil
	}

	// Array of content blocks — may contain text and/or images.
	var inner []AnthropicContentBlock
	if err := json.Unmarshal(b.Content, &inner); err != nil {
		return "(empty)", nil
	}

	// Separate text (for function_call_output) from images (for user message).
	var textParts []string
	var imageParts []ResponsesContentPart
	for _, ib := range inner {
		switch ib.Type {
		case "text":
			if ib.Text != "" {
				textParts = append(textParts, ib.Text)
			}
		case "image":
			if uri := anthropicImageToDataURI(ib.Source); uri != "" {
				imageParts = append(imageParts, ResponsesContentPart{Type: "input_image", ImageURL: uri})
			}
		}
	}

	text := strings.Join(textParts, "\n\n")
	if text == "" {
		text = "(empty)"
	}
	return text, imageParts
}

// mapAnthropicEffortToResponses converts Anthropic reasoning effort levels to
// OpenAI Responses API effort levels.
//
// Both APIs default to "high". The mapping is 1:1 for shared levels;
// only Anthropic's "max" (Opus 4.6 exclusive) maps to OpenAI's "xhigh"
// (GPT-5.2+ exclusive) as both represent the highest reasoning tier.
//
//	low    → low
//	medium → medium
//	high   → high
//	max    → xhigh
func mapAnthropicEffortToResponses(effort string) string {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "max":
		return "xhigh"
	case "low", "medium", "high":
		return strings.ToLower(strings.TrimSpace(effort))
	default:
		return "high"
	}
}

// convertAnthropicToolsToResponses maps Anthropic tool definitions to
// Responses API tools. Server-side tools like web_search are mapped to their
// OpenAI equivalents; MCP toolsets are reconstructed from mcp_servers.
func convertAnthropicToolsToResponses(tools []AnthropicTool, mcpServers []AnthropicMCPServer) []ResponsesTool {
	var out []ResponsesTool
	mcpServerByName := make(map[string]AnthropicMCPServer, len(mcpServers))
	for _, server := range mcpServers {
		if server.Name != "" {
			mcpServerByName[server.Name] = server
		}
	}

	for _, t := range tools {
		if strings.HasPrefix(t.Type, "web_search") {
			out = append(out, ResponsesTool{Type: "web_search"})
			continue
		}
		if strings.HasPrefix(t.Type, "code_execution") {
			out = append(out, ResponsesTool{Type: "code_interpreter", Name: t.Name, Parameters: t.InputSchema})
			continue
		}
		if t.Type == "mcp_toolset" {
			tool := ResponsesTool{
				Type:        "mcp",
				ServerLabel: t.MCPServerName,
				Parameters:  buildResponsesMCPToolParameters(t),
			}
			if server, ok := mcpServerByName[t.MCPServerName]; ok {
				tool.ServerURL = server.URL
				tool.Authorization = server.AuthorizationToken
				if tool.ServerLabel == "" {
					tool.ServerLabel = server.Name
				}
			}
			if allowedTools := extractAllowedToolsFromAnthropicMCPToolset(t); len(allowedTools) > 0 {
				allowedJSON, _ := json.Marshal(allowedTools)
				tool.AllowedTools = allowedJSON
			}
			out = append(out, tool)
			continue
		}
		out = append(out, ResponsesTool{
			Type:        "function",
			Name:        t.Name,
			Description: t.Description,
			Parameters:  normalizeToolParameters(t.InputSchema),
		})
	}

	for _, server := range mcpServers {
		if !hasAnthropicMCPToolset(tools, server.Name) {
			out = append(out, ResponsesTool{
				Type:          "mcp",
				ServerLabel:   server.Name,
				ServerURL:     server.URL,
				Authorization: server.AuthorizationToken,
			})
		}
	}

	return out
}

func hasAnthropicMCPToolset(tools []AnthropicTool, serverName string) bool {
	for _, tool := range tools {
		if tool.Type == "mcp_toolset" && tool.MCPServerName == serverName {
			return true
		}
	}
	return false
}

func extractAllowedToolsFromAnthropicMCPToolset(tool AnthropicTool) []string {
	var defaultConfig struct {
		Enabled *bool `json:"enabled,omitempty"`
	}
	if len(tool.DefaultConfig) > 0 {
		_ = json.Unmarshal(tool.DefaultConfig, &defaultConfig)
	}

	var legacy struct {
		AllowedTools []string `json:"allowed_tools,omitempty"`
	}
	if len(tool.DefaultConfig) > 0 {
		_ = json.Unmarshal(tool.DefaultConfig, &legacy)
		if len(legacy.AllowedTools) > 0 {
			return legacy.AllowedTools
		}
	}

	var configs map[string]struct {
		Enabled *bool `json:"enabled,omitempty"`
	}
	if len(tool.Configs) == 0 || json.Unmarshal(tool.Configs, &configs) != nil {
		return nil
	}
	if defaultConfig.Enabled == nil || *defaultConfig.Enabled {
		return nil
	}
	allowed := make([]string, 0, len(configs))
	for name, cfg := range configs {
		if cfg.Enabled != nil && *cfg.Enabled {
			allowed = append(allowed, name)
		}
	}
	return allowed
}

func buildResponsesMCPToolParameters(tool AnthropicTool) json.RawMessage {
	payload := map[string]json.RawMessage{}
	if len(tool.DefaultConfig) > 0 {
		payload["default_config"] = tool.DefaultConfig
	}
	if len(tool.Configs) > 0 {
		payload["configs"] = tool.Configs
	}
	if len(tool.CacheControl) > 0 {
		payload["cache_control"] = tool.CacheControl
	}
	if len(payload) == 0 {
		return nil
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil
	}
	return encoded
}

// normalizeToolParameters ensures the tool parameter schema is valid for
// OpenAI's Responses API, which requires "properties" on object schemas.
//
//   - nil/empty → {"type":"object","properties":{}}
//   - type=object without properties → adds "properties": {}
//   - otherwise → returned unchanged
func normalizeToolParameters(schema json.RawMessage) json.RawMessage {
	if len(schema) == 0 || string(schema) == "null" {
		return json.RawMessage(`{"type":"object","properties":{}}`)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(schema, &m); err != nil {
		return schema
	}

	typ := m["type"]
	if string(typ) != `"object"` {
		return schema
	}

	if _, ok := m["properties"]; ok {
		return schema
	}

	m["properties"] = json.RawMessage(`{}`)
	out, err := json.Marshal(m)
	if err != nil {
		return schema
	}
	return out
}
