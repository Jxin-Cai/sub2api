package apicompat

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ResponsesToAnthropicRequest converts a Responses API request into an
// Anthropic Messages request. This is the reverse of AnthropicToResponses and
// enables Anthropic platform groups to accept OpenAI Responses API requests
// by converting them to the native /v1/messages format before forwarding upstream.
func ResponsesToAnthropicRequest(req *ResponsesRequest) (*AnthropicRequest, error) {
	system, messages, err := convertResponsesInputToAnthropic(req.Instructions, req.Input)
	if err != nil {
		return nil, err
	}

	out := &AnthropicRequest{
		Model:       req.Model,
		Messages:    messages,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
	}

	if len(req.Metadata) > 0 {
		out.Metadata = req.Metadata
	}
	if req.ServiceTier != "" {
		out.ServiceTier = req.ServiceTier
	}
	if len(req.ContextManagement) > 0 {
		out.ContextManagement = responsesContextManagementToAnthropic(req.ContextManagement)
	}
	if len(req.Conversation) > 0 {
		out.Container = req.Conversation
	}

	if len(system) > 0 {
		out.System = system
	}

	if req.MaxOutputTokens != nil && *req.MaxOutputTokens > 0 {
		out.MaxTokens = *req.MaxOutputTokens
	}
	if out.MaxTokens == 0 {
		out.MaxTokens = 8192
	}

	if len(req.Tools) > 0 {
		anthTools, mcpServers := convertResponsesToAnthropicTools(req.Tools)
		out.Tools = anthTools
		out.MCPServers = mcpServers
	}

	if len(req.ToolChoice) > 0 {
		tc, err := convertResponsesToAnthropicToolChoice(req.ToolChoice)
		if err != nil {
			return nil, fmt.Errorf("convert tool_choice: %w", err)
		}
		out.ToolChoice = tc
	}
	if req.ParallelToolCalls != nil && !*req.ParallelToolCalls {
		mergedToolChoice, err := mergeDisableParallelToolUse(out.ToolChoice)
		if err != nil {
			return nil, fmt.Errorf("merge tool_choice disable_parallel_tool_use: %w", err)
		}
		out.ToolChoice = mergedToolChoice
	}

	if req.Reasoning != nil && req.Reasoning.Effort != "" {
		effort := normalizeResponsesReasoningEffort(req.Reasoning.Effort)
		if effort != "" && effort != "none" {
			anthropicEffort := mapResponsesEffortToAnthropic(effort)
			out.OutputConfig = &AnthropicOutputConfig{Effort: anthropicEffort}
			if anthropicEffort != "low" {
				out.Thinking = &AnthropicThinking{
					Type:         "enabled",
					BudgetTokens: defaultThinkingBudget(anthropicEffort),
				}
			}
		}
	}

	if format := responsesTextFormat(req.Text); len(format) > 0 {
		if out.OutputConfig == nil {
			out.OutputConfig = &AnthropicOutputConfig{}
		}
		out.OutputConfig.Format = format
	}

	return out, nil
}

// defaultThinkingBudget returns a sensible thinking budget based on effort level.
func defaultThinkingBudget(effort string) int {
	switch effort {
	case "low":
		return 1024
	case "medium":
		return 4096
	case "high":
		return 10240
	case "max":
		return 32768
	default:
		return 10240
	}
}

// mapResponsesEffortToAnthropic converts OpenAI Responses reasoning effort to
// Anthropic effort levels. Reverse of mapAnthropicEffortToResponses.
func mapResponsesEffortToAnthropic(effort string) string {
	if effort == "xhigh" {
		return "max"
	}
	return effort
}

func normalizeResponsesReasoningEffort(effort string) string {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "minimal", "none":
		return "none"
	case "low", "medium", "high", "xhigh":
		return strings.ToLower(strings.TrimSpace(effort))
	default:
		return ""
	}
}

// convertResponsesInputToAnthropic extracts system prompt and messages from
// a Responses API instructions + input array. Returns the system as raw JSON
// (for Anthropic's polymorphic system field) and a list of Anthropic messages.
func convertResponsesInputToAnthropic(instructions string, inputRaw json.RawMessage) (json.RawMessage, []AnthropicMessage, error) {
	var systemParts []string
	if strings.TrimSpace(instructions) != "" {
		systemParts = append(systemParts, strings.TrimSpace(instructions))
	}

	// Try as plain string input.
	var inputStr string
	if err := json.Unmarshal(inputRaw, &inputStr); err == nil {
		content, _ := json.Marshal(inputStr)
		var system json.RawMessage
		if len(systemParts) > 0 {
			system, _ = json.Marshal(strings.Join(systemParts, "\n\n"))
		}
		return system, []AnthropicMessage{{Role: "user", Content: content}}, nil
	}

	var items []ResponsesInputItem
	if err := json.Unmarshal(inputRaw, &items); err != nil {
		return nil, nil, fmt.Errorf("parse responses input: %w", err)
	}

	var messages []AnthropicMessage

	for _, item := range items {
		switch {
		case item.Role == "system" || item.Role == "developer":
			text := extractTextFromContent(item.Content)
			if text != "" {
				systemParts = append(systemParts, text)
			}

		case item.Type == "function_call":
			input := json.RawMessage("{}")
			if item.Arguments != "" {
				input = json.RawMessage(item.Arguments)
			}
			blockType := "tool_use"
			if isResponsesMCPNamespace(item.Namespace, item.RawItem) {
				blockType = "mcp_tool_use"
			}
			block := AnthropicContentBlock{
				Type:       blockType,
				ID:         fromResponsesCallIDToAnthropic(item.CallID),
				Name:       item.Name,
				ServerName: item.Namespace,
				Input:      input,
			}
			blockJSON, _ := json.Marshal([]AnthropicContentBlock{block})
			messages = append(messages, AnthropicMessage{Role: "assistant", Content: blockJSON})

		case item.Type == "function_call_output":
			contentJSON := item.OutputRaw
			if len(contentJSON) == 0 {
				outputContent := item.Output
				if outputContent == "" {
					outputContent = "(empty)"
				}
				contentJSON, _ = json.Marshal(outputContent)
			}
			blockType := "tool_result"
			if isResponsesMCPNamespace(item.Namespace, item.RawItem) {
				blockType = "mcp_tool_result"
			}
			block := AnthropicContentBlock{
				Type:      blockType,
				ToolUseID: fromResponsesCallIDToAnthropic(item.CallID),
				Content:   contentJSON,
				IsError:   item.Status == "incomplete",
			}
			blockJSON, _ := json.Marshal([]AnthropicContentBlock{block})
			messages = append(messages, AnthropicMessage{Role: "user", Content: blockJSON})

		case item.Role == "user":
			content, err := convertResponsesUserToAnthropicContent(item.Content)
			if err != nil {
				return nil, nil, err
			}
			messages = append(messages, AnthropicMessage{Role: "user", Content: content})

		case item.Role == "assistant":
			content, err := convertResponsesAssistantToAnthropicContent(item.Content)
			if err != nil {
				return nil, nil, err
			}
			messages = append(messages, AnthropicMessage{Role: "assistant", Content: content})

		default:
			if item.Content != nil {
				messages = append(messages, AnthropicMessage{Role: "user", Content: item.Content})
			}
		}
	}

	// Repair tool_use/tool_result pairing, then merge consecutive same-role
	// messages (Anthropic requires alternating roles). The first merge groups
	// parallel calls (and their results) so the pairing pass sees them together;
	// the pairing pass may re-split a user turn (e.g. when an injected message
	// sat between a call and its output), so a second merge restores alternation.
	messages = mergeConsecutiveMessages(messages)
	messages = normalizeAnthropicToolPairing(messages)
	messages = mergeConsecutiveMessages(messages)

	var system json.RawMessage
	if len(systemParts) > 0 {
		system, _ = json.Marshal(strings.Join(systemParts, "\n\n"))
	}

	return system, messages, nil
}

func responsesContextManagementToAnthropic(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	var edits []json.RawMessage
	if err := json.Unmarshal(raw, &edits); err == nil {
		return responsesContextManagementEditsToAnthropic(edits)
	}
	var payload struct {
		Edits []json.RawMessage `json:"edits,omitempty"`
	}
	if err := json.Unmarshal(raw, &payload); err == nil {
		return responsesContextManagementEditsToAnthropic(payload.Edits)
	}
	return nil
}

func responsesContextManagementEditsToAnthropic(edits []json.RawMessage) json.RawMessage {
	if len(edits) == 0 {
		return nil
	}
	converted := make([]json.RawMessage, 0, len(edits))
	for _, edit := range edits {
		converted = append(converted, responsesContextManagementEditToAnthropic(edit))
	}
	encoded, err := json.Marshal(map[string]any{"edits": converted})
	if err == nil {
		return encoded
	}
	return nil
}

func responsesContextManagementEditToAnthropic(raw json.RawMessage) json.RawMessage {
	var edit map[string]any
	if err := json.Unmarshal(raw, &edit); err != nil {
		return raw
	}
	if strings.TrimSpace(stringValue(edit["type"])) != "compaction" {
		return raw
	}
	converted := map[string]any{"type": "compact_20260112"}
	if threshold, ok := intValue(edit["compact_threshold"]); ok {
		converted["trigger"] = map[string]any{"type": "input_tokens", "value": threshold}
	}
	return mustMarshalRawJSON(converted)
}

func responsesTextFormat(text *ResponsesText) json.RawMessage {
	if text == nil || len(text.Format) == 0 {
		return nil
	}
	return text.Format
}

func extractResponsesTextFormat(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil
	}
	format := payload["format"]
	if len(format) == 0 || string(format) == "null" {
		return nil
	}
	return format
}

func isResponsesMCPNamespace(namespace string, raw json.RawMessage) bool {
	if namespace != "" && namespace != "mcp" {
		return true
	}
	if len(raw) == 0 {
		return false
	}
	var item struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &item); err != nil {
		return false
	}
	return item.Type == "mcp_tool_use" || item.Type == "mcp_tool_result"
}

// normalizeAnthropicToolPairing rebuilds the message sequence so it satisfies
// Anthropic's tool_use/tool_result invariants, which the naive item-by-item
// conversion violates whenever the Responses history interleaves anything
// between a function_call and its function_call_output:
//
//   - every tool_result block must have a matching tool_use in the immediately
//     preceding assistant message ("tool_result ... must have a corresponding
//     tool_use block in the previous message");
//   - every tool_use block must be answered by a tool_result in the immediately
//     following user message (Anthropic rejects unanswered tool_use ids);
//   - user/assistant turns must alternate.
//
// codex (Responses, store:false) re-sends the whole history each turn and
// frequently injects items between a call and its output — a developer/approval
// notice, or a sibling parallel call whose output never arrived. The unrepaired
// converter emits each function_call as its own assistant message and each
// output as its own user message, so any such interleaving breaks
// tool_use↔tool_result adjacency and yields an upstream 400.
//
// The repair indexes every tool_result by its tool_use id, then for each
// assistant message carrying tool_use blocks keeps only the answered ones
// (dropping unanswered/dangling calls — and the assistant message entirely if it
// has no other content) and emits the matching tool_result blocks, in call
// order, as the very next user message. Standalone tool_result blocks are
// dropped from their original position (re-emitted adjacent to their call);
// orphan tool_results with no announcing tool_use are dropped. Non-tool content
// passes through in place. This mirrors normalizeChatMessages on the
// Responses→Chat path.
func normalizeAnthropicToolPairing(messages []AnthropicMessage) []AnthropicMessage {
	// Index every tool_result block by its tool_use id (last wins on dup).
	results := make(map[string]AnthropicContentBlock)
	for _, m := range messages {
		if m.Role != "user" {
			continue
		}
		for _, b := range parseContentBlocks(m.Content) {
			if b.Type == "tool_result" && b.ToolUseID != "" {
				results[b.ToolUseID] = b
			}
		}
	}

	out := make([]AnthropicMessage, 0, len(messages))
	for _, m := range messages {
		blocks := parseContentBlocks(m.Content)
		switch m.Role {
		case "assistant":
			var toolUses, others []AnthropicContentBlock
			for _, b := range blocks {
				if b.Type == "tool_use" {
					toolUses = append(toolUses, b)
				} else {
					others = append(others, b)
				}
			}
			if len(toolUses) == 0 {
				out = append(out, m)
				continue
			}
			kept := make([]AnthropicContentBlock, 0, len(toolUses))
			for _, tu := range toolUses {
				if _, ok := results[tu.ID]; ok {
					kept = append(kept, tu)
				}
			}
			if len(kept) == 0 {
				// No answered calls: keep any non-tool content, else drop.
				if len(others) > 0 {
					out = append(out, anthropicMessageFromBlocks("assistant", others))
				}
				continue
			}
			asstBlocks := make([]AnthropicContentBlock, 0, len(others)+len(kept))
			asstBlocks = append(asstBlocks, others...)
			asstBlocks = append(asstBlocks, kept...)
			out = append(out, anthropicMessageFromBlocks("assistant", asstBlocks))

			resBlocks := make([]AnthropicContentBlock, 0, len(kept))
			for _, tu := range kept {
				resBlocks = append(resBlocks, results[tu.ID])
			}
			out = append(out, anthropicMessageFromBlocks("user", resBlocks))

		case "user":
			var nonResult []AnthropicContentBlock
			hasResult := false
			for _, b := range blocks {
				if b.Type == "tool_result" {
					hasResult = true
					continue
				}
				nonResult = append(nonResult, b)
			}
			if !hasResult {
				out = append(out, m)
				continue
			}
			// The tool_result blocks are re-emitted next to their call; keep any
			// other content of this user turn in place, drop it if there is none.
			if len(nonResult) > 0 {
				out = append(out, anthropicMessageFromBlocks("user", nonResult))
			}

		default:
			out = append(out, m)
		}
	}
	return out
}

// anthropicMessageFromBlocks builds an AnthropicMessage whose content is the
// marshaled block array.
func anthropicMessageFromBlocks(role string, blocks []AnthropicContentBlock) AnthropicMessage {
	content, _ := json.Marshal(blocks)
	return AnthropicMessage{Role: role, Content: content}
}

// extractTextFromContent extracts text from a content field that may be a
// plain string or an array of content parts.
func extractTextFromContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []ResponsesContentPart
	if err := json.Unmarshal(raw, &parts); err == nil {
		var texts []string
		for _, p := range parts {
			if (p.Type == "input_text" || p.Type == "output_text" || p.Type == "text") && p.Text != "" {
				texts = append(texts, p.Text)
			}
		}
		return strings.Join(texts, "\n\n")
	}
	return ""
}

// convertResponsesUserToAnthropicContent converts a Responses user message
// content field into Anthropic content blocks JSON.
func convertResponsesUserToAnthropicContent(raw json.RawMessage) (json.RawMessage, error) {
	if len(raw) == 0 {
		return json.Marshal("")
	}

	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return json.Marshal(s)
	}

	var parts []ResponsesContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return raw, nil
	}

	var blocks []AnthropicContentBlock
	for _, p := range parts {
		switch p.Type {
		case "input_text", "text":
			if p.Text != "" {
				blocks = append(blocks, AnthropicContentBlock{Type: "text", Text: p.Text})
			}
		case "input_image":
			src := dataURIToAnthropicImageSource(p.ImageURL)
			if src != nil {
				blocks = append(blocks, AnthropicContentBlock{Type: "image", Source: src})
			}
		case "input_audio":
			if text := describeResponsesAudioPart(p); text != "" {
				blocks = append(blocks, AnthropicContentBlock{Type: "text", Text: text})
			}
		case "file", "input_file":
			if text := describeResponsesFilePart(p); text != "" {
				blocks = append(blocks, AnthropicContentBlock{Type: "text", Text: text})
			}
		case "file_data", "file_id", "file_url":
			if text := describeResponsesFilePart(p); text != "" {
				blocks = append(blocks, AnthropicContentBlock{Type: "text", Text: text})
			}
		}
	}

	if len(blocks) == 0 {
		return json.Marshal("")
	}
	return json.Marshal(blocks)
}

// convertResponsesAssistantToAnthropicContent converts a Responses assistant
// message content field into Anthropic content blocks JSON.
func convertResponsesAssistantToAnthropicContent(raw json.RawMessage) (json.RawMessage, error) {
	if len(raw) == 0 {
		return json.Marshal([]AnthropicContentBlock{{Type: "text", Text: ""}})
	}

	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return json.Marshal([]AnthropicContentBlock{{Type: "text", Text: s}})
	}

	var parts []ResponsesContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return raw, nil
	}

	var blocks []AnthropicContentBlock
	for _, p := range parts {
		switch p.Type {
		case "output_text", "text":
			if p.Text != "" {
				blocks = append(blocks, AnthropicContentBlock{Type: "text", Text: p.Text})
			}
		}
	}

	if len(blocks) == 0 {
		blocks = append(blocks, AnthropicContentBlock{Type: "text", Text: ""})
	}
	return json.Marshal(blocks)
}

// fromResponsesCallIDToAnthropic converts an OpenAI function call ID back to Anthropic format.
func fromResponsesCallIDToAnthropic(id string) string {
	return id
}

// dataURIToAnthropicImageSource parses a data URI into an AnthropicImageSource.
func dataURIToAnthropicImageSource(dataURI string) *AnthropicImageSource {
	if !strings.HasPrefix(dataURI, "data:") {
		return nil
	}
	rest := strings.TrimPrefix(dataURI, "data:")
	semicolonIdx := strings.Index(rest, ";")
	if semicolonIdx < 0 {
		return nil
	}
	mediaType := rest[:semicolonIdx]
	rest = rest[semicolonIdx+1:]
	if !strings.HasPrefix(rest, "base64,") {
		return nil
	}
	data := strings.TrimPrefix(rest, "base64,")
	return &AnthropicImageSource{Type: "base64", MediaType: mediaType, Data: data}
}

func describeResponsesAudioPart(part ResponsesContentPart) string {
	if part.InputAudio == nil {
		return ""
	}
	format := strings.TrimSpace(part.InputAudio.Format)
	if format == "" {
		format = "unknown"
	}
	return fmt.Sprintf("[audio input omitted in Anthropic conversion: format=%s]", format)
}

func describeResponsesFilePart(part ResponsesContentPart) string {
	parts := make([]string, 0, 4)
	if part.FileID != "" {
		parts = append(parts, "file_id="+part.FileID)
	}
	if part.FileURL != "" {
		parts = append(parts, "file_url="+part.FileURL)
	}
	if part.Filename != "" {
		parts = append(parts, "filename="+part.Filename)
	}
	if part.FileData != "" {
		parts = append(parts, "file_data=inline")
	}
	if len(parts) == 0 {
		return ""
	}
	return "[file input omitted in Anthropic conversion: " + strings.Join(parts, ", ") + "]"
}

func stringValue(value any) string {
	str, _ := value.(string)
	return str
}

func intValue(value any) (int, bool) {
	switch v := value.(type) {
	case float64:
		return int(v), true
	case int:
		return v, true
	case int32:
		return int(v), true
	case int64:
		return int(v), true
	default:
		return 0, false
	}
}

func mustMarshalRawJSON(value any) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	return encoded
}

// mergeConsecutiveMessages merges consecutive messages with the same role because Anthropic requires alternating turns.
func mergeConsecutiveMessages(messages []AnthropicMessage) []AnthropicMessage {
	if len(messages) <= 1 {
		return messages
	}

	var merged []AnthropicMessage
	for _, msg := range messages {
		if len(merged) == 0 || merged[len(merged)-1].Role != msg.Role {
			merged = append(merged, msg)
			continue
		}

		last := &merged[len(merged)-1]
		lastBlocks := parseContentBlocks(last.Content)
		newBlocks := parseContentBlocks(msg.Content)
		combined := append(lastBlocks, newBlocks...)
		last.Content, _ = json.Marshal(combined)
	}
	return merged
}

// parseContentBlocks attempts to parse content as []AnthropicContentBlock.
func parseContentBlocks(raw json.RawMessage) []AnthropicContentBlock {
	var blocks []AnthropicContentBlock
	if err := json.Unmarshal(raw, &blocks); err == nil {
		return blocks
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []AnthropicContentBlock{{Type: "text", Text: s}}
	}
	return nil
}

// convertResponsesToAnthropicTools maps Responses API tools to Anthropic format.
func convertResponsesToAnthropicTools(tools []ResponsesTool) ([]AnthropicTool, []AnthropicMCPServer) {
	var out []AnthropicTool
	var mcpServers []AnthropicMCPServer
	for _, t := range tools {
		switch t.Type {
		case "web_search", "google_search", "web_search_20250305":
			out = append(out, AnthropicTool{
				Type: "web_search_20250305",
				Name: "web_search",
			})
		case "code_interpreter", "code_execution":
			out = append(out, AnthropicTool{Type: "code_execution_20260120", Name: t.Name, InputSchema: t.Parameters})
		case "mcp":
			server, toolset := convertResponsesMCPToolToAnthropic(t)
			mcpServers = append(mcpServers, server)
			out = append(out, toolset)
		case "function":
			out = append(out, AnthropicTool{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: normalizeAnthropicInputSchema(t.Parameters),
			})
		case "custom":
			out = append(out, AnthropicTool{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: normalizeAnthropicInputSchema(t.Parameters),
			})
		default:
			out = append(out, AnthropicTool{
				Type:        t.Type,
				Name:        t.Name,
				Description: t.Description,
				InputSchema: normalizeAnthropicInputSchema(t.Parameters),
			})
		}
	}
	return out, mcpServers
}

func convertResponsesMCPToolToAnthropic(tool ResponsesTool) (AnthropicMCPServer, AnthropicTool) {
	serverName := strings.TrimSpace(tool.ServerLabel)
	if serverName == "" {
		serverName = strings.TrimSpace(tool.ConnectorID)
	}
	if serverName == "" {
		serverName = "mcp-server"
	}

	server := AnthropicMCPServer{
		Type:               "url",
		URL:                tool.ServerURL,
		Name:               serverName,
		AuthorizationToken: tool.Authorization,
	}
	toolset := AnthropicTool{Type: "mcp_toolset", MCPServerName: serverName}

	if len(tool.Parameters) > 0 {
		var params map[string]json.RawMessage
		if json.Unmarshal(tool.Parameters, &params) == nil {
			toolset.DefaultConfig = params["default_config"]
			toolset.Configs = params["configs"]
			toolset.CacheControl = params["cache_control"]
		}
	}

	if len(tool.AllowedTools) > 0 && len(toolset.Configs) == 0 {
		var allowed []string
		if json.Unmarshal(tool.AllowedTools, &allowed) == nil && len(allowed) > 0 {
			toolset.DefaultConfig = json.RawMessage(`{"enabled":false}`)
			configs := make(map[string]map[string]bool, len(allowed))
			for _, name := range allowed {
				configs[name] = map[string]bool{"enabled": true}
			}
			toolset.Configs, _ = json.Marshal(configs)
		}
	}

	return server, toolset
}

// normalizeAnthropicInputSchema ensures input_schema is a valid object schema.
func normalizeAnthropicInputSchema(schema json.RawMessage) json.RawMessage {
	const emptyObjectSchema = `{"type":"object","properties":{}}`

	trimmed := strings.TrimSpace(string(schema))
	if trimmed == "" || trimmed == "null" {
		return json.RawMessage(emptyObjectSchema)
	}

	var m map[string]json.RawMessage
	if err := json.Unmarshal(schema, &m); err != nil {
		return json.RawMessage(`{"type":"object","properties":{}}`)
	}

	typeRaw, ok := m["type"]
	if !ok || strings.TrimSpace(string(typeRaw)) == "" || string(typeRaw) == "null" {
		m["type"] = json.RawMessage(`"object"`)
	} else {
		var typ string
		if err := json.Unmarshal(typeRaw, &typ); err != nil || typ != "object" {
			return json.RawMessage(emptyObjectSchema)
		}
	}

	if _, ok := m["properties"]; !ok {
		m["properties"] = json.RawMessage(`{}`)
	}

	out, err := json.Marshal(m)
	if err != nil {
		return json.RawMessage(emptyObjectSchema)
	}
	return out
}

// convertResponsesToAnthropicToolChoice maps Responses tool_choice to Anthropic format.
// Reverse of convertAnthropicToolChoiceToResponses.
//
//	"auto"                                     → {"type":"auto"}
//	"required"                                 → {"type":"any"}
//	"none"                                     → {"type":"none"}
//	{"type":"function","name":"X"}                 → {"type":"tool","name":"X"}
//	{"type":"function","function":{"name":"X"}}     → {"type":"tool","name":"X"} // legacy
func convertResponsesToAnthropicToolChoice(raw json.RawMessage) (json.RawMessage, error) {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		switch s {
		case "auto":
			return json.Marshal(map[string]string{"type": "auto"})
		case "required":
			return json.Marshal(map[string]string{"type": "any"})
		case "none":
			return json.Marshal(map[string]string{"type": "none"})
		default:
			return raw, nil
		}
	}

	var tc struct {
		Type     string `json:"type"`
		Name     string `json:"name"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &tc); err == nil && tc.Type == "function" {
		name := strings.TrimSpace(tc.Name)
		if name == "" {
			name = strings.TrimSpace(tc.Function.Name)
		}
		if name == "" {
			return raw, nil
		}
		return json.Marshal(map[string]string{
			"type": "tool",
			"name": name,
		})
	}

	return raw, nil
}

func mergeDisableParallelToolUse(raw json.RawMessage) (json.RawMessage, error) {
	if len(raw) == 0 {
		return json.Marshal(map[string]any{"type": "auto", "disable_parallel_tool_use": true})
	}
	var tc map[string]any
	if err := json.Unmarshal(raw, &tc); err != nil {
		return nil, err
	}
	tc["disable_parallel_tool_use"] = true
	return json.Marshal(tc)
}
