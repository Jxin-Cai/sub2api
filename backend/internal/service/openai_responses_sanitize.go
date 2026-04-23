package service

import (
	"encoding/json"
	"strings"
)

var openAIResponsesRequestAllowedKeys = map[string]struct{}{
	"background":             {},
	"context_management":     {},
	"conversation":           {},
	"include":                {},
	"input":                  {},
	"instructions":           {},
	"max_output_tokens":      {},
	"max_tool_calls":         {},
	"metadata":               {},
	"model":                  {},
	"parallel_tool_calls":    {},
	"previous_response_id":   {},
	"prompt":                 {},
	"prompt_cache_key":       {},
	"prompt_cache_retention": {},
	"reasoning":              {},
	"safety_identifier":      {},
	"service_tier":           {},
	"store":                  {},
	"stream":                 {},
	"stream_options":         {},
	"temperature":            {},
	"text":                   {},
	"tool_choice":            {},
	"tools":                  {},
	"top_logprobs":           {},
	"top_p":                  {},
	"truncation":             {},
	"user":                   {},
}

var openAIResponsesKnownInputItemTypes = map[string]map[string]struct{}{
	"message": {
		"type": {}, "id": {}, "role": {}, "content": {}, "phase": {}, "status": {},
	},
	"function_call": {
		"type": {}, "id": {}, "call_id": {}, "name": {}, "arguments": {}, "status": {},
	},
	"function_call_output": {
		"type": {}, "id": {}, "call_id": {}, "output": {}, "status": {},
	},
	"reasoning": {
		"type": {}, "id": {}, "summary": {}, "encrypted_content": {},
	},
	"compaction": {
		"type": {}, "id": {}, "encrypted_content": {},
	},
	"item_reference": {
		"type": {}, "id": {},
	},
}

var openAIResponsesRoleInputItemAllowedKeys = map[string]struct{}{
	"type": {}, "id": {}, "role": {}, "content": {}, "phase": {}, "status": {},
}

var openAIResponsesReasoningAllowedKeys = map[string]struct{}{
	"effort": {}, "summary": {}, "generate_summary": {}, "encrypted_content": {},
}

var openAIResponsesContextManagementAllowedKeys = map[string]struct{}{
	"type":              {},
	"compact_threshold": {},
}

var openAIResponsesContextManagementAllowedTypes = map[string]struct{}{
	"clear_function_results": {},
	"compaction":             {},
}

var openAIResponsesCompatOnlyInputKeys = []string{"output_raw", "namespace", "item", "raw_item"}

func sanitizeOpenAIResponsesRequestBody(body []byte) ([]byte, bool, error) {
	if len(body) == 0 {
		return body, false, nil
	}

	var reqBody map[string]any
	if err := json.Unmarshal(body, &reqBody); err != nil {
		return body, false, nil
	}
	if !sanitizeOpenAIResponsesRequestMap(reqBody) {
		return body, false, nil
	}
	updated, err := json.Marshal(reqBody)
	if err != nil {
		return nil, false, err
	}
	return updated, true, nil
}

func sanitizeOpenAIResponsesRequestMap(reqBody map[string]any) bool {
	if len(reqBody) == 0 {
		return false
	}

	modified := false
	for key := range reqBody {
		if _, ok := openAIResponsesRequestAllowedKeys[key]; ok {
			continue
		}
		delete(reqBody, key)
		modified = true
	}

	if normalizedContextManagement, changed, keep := sanitizeOpenAIResponsesContextManagement(reqBody["context_management"]); changed {
		modified = true
		if keep {
			reqBody["context_management"] = normalizedContextManagement
		} else {
			delete(reqBody, "context_management")
		}
	}

	if normalizedMaxToolCalls, changed, keep := sanitizeOpenAIResponsesMaxToolCalls(reqBody["max_tool_calls"]); changed {
		modified = true
		if keep {
			reqBody["max_tool_calls"] = normalizedMaxToolCalls
		} else {
			delete(reqBody, "max_tool_calls")
		}
	}

	if input, ok := reqBody["input"].([]any); ok {
		sanitized, inputModified := sanitizeOpenAIResponsesInput(input)
		if inputModified {
			reqBody["input"] = sanitized
			modified = true
		}
	}

	if reasoning, ok := reqBody["reasoning"].(map[string]any); ok {
		if sanitizeOpenAIResponsesReasoning(reasoning) {
			modified = true
		}
		if len(reasoning) == 0 {
			delete(reqBody, "reasoning")
			modified = true
		}
	}

	if sanitizedToolChoice, changed, keep := sanitizeOpenAIResponsesToolChoice(reqBody["tool_choice"]); changed {
		modified = true
		if keep {
			reqBody["tool_choice"] = sanitizedToolChoice
		} else {
			delete(reqBody, "tool_choice")
		}
	}

	return modified
}

func sanitizeOpenAIResponsesMaxToolCalls(value any) (any, bool, bool) {
	if value == nil {
		return nil, false, false
	}
	count, ok := intValueFromAny(value)
	if !ok || count <= 0 {
		return nil, true, false
	}
	if value == count {
		return value, false, true
	}
	return count, true, true
}

func sanitizeOpenAIResponsesInput(input []any) ([]any, bool) {
	filtered := make([]any, 0, len(input))
	modified := false
	for _, rawItem := range input {
		item, ok := rawItem.(map[string]any)
		if !ok {
			filtered = append(filtered, rawItem)
			continue
		}
		sanitizedItem, itemModified := sanitizeOpenAIResponsesInputItem(item)
		if itemModified {
			modified = true
		}
		filtered = append(filtered, sanitizedItem)
	}
	return filtered, modified
}

func sanitizeOpenAIResponsesInputItem(item map[string]any) (map[string]any, bool) {
	newItem := make(map[string]any, len(item))
	for key, value := range item {
		newItem[key] = value
	}
	modified := false

	for _, compatKey := range openAIResponsesCompatOnlyInputKeys {
		if _, ok := newItem[compatKey]; ok {
			delete(newItem, compatKey)
			modified = true
		}
	}

	itemType := strings.TrimSpace(stringValueFromAny(newItem["type"]))
	if allowed, ok := openAIResponsesKnownInputItemTypes[itemType]; ok {
		for key := range newItem {
			if _, keep := allowed[key]; keep {
				continue
			}
			delete(newItem, key)
			modified = true
		}
		return newItem, modified
	}

	if role := strings.TrimSpace(stringValueFromAny(newItem["role"])); role != "" {
		for key := range newItem {
			if _, keep := openAIResponsesRoleInputItemAllowedKeys[key]; keep {
				continue
			}
			delete(newItem, key)
			modified = true
		}
	}

	return newItem, modified
}

func sanitizeOpenAIResponsesContextManagement(value any) (any, bool, bool) {
	if value == nil {
		return nil, false, false
	}

	switch v := value.(type) {
	case []any:
		filtered := make([]any, 0, len(v))
		modified := false
		for _, raw := range v {
			item, ok := raw.(map[string]any)
			if !ok {
				modified = true
				continue
			}
			normalizedItem, itemModified, keep := sanitizeOpenAIResponsesContextManagementItem(item)
			if itemModified {
				modified = true
			}
			if keep {
				filtered = append(filtered, normalizedItem)
			} else {
				modified = true
			}
		}
		if len(filtered) == 0 {
			return nil, true, false
		}
		if !modified && len(filtered) == len(v) {
			return value, false, true
		}
		return filtered, true, true
	case map[string]any:
		if edits, ok := v["edits"].([]any); ok {
			return sanitizeOpenAIResponsesContextManagement(edits)
		}
		return nil, true, false
	default:
		return nil, true, false
	}
}

func sanitizeOpenAIResponsesContextManagementItem(item map[string]any) (map[string]any, bool, bool) {
	normalized := make(map[string]any, len(item))
	modified := false
	for key, value := range item {
		if _, ok := openAIResponsesContextManagementAllowedKeys[key]; !ok {
			modified = true
			continue
		}
		normalized[key] = value
	}
	typ := strings.TrimSpace(stringValueFromAny(normalized["type"]))
	if typ == "" {
		return nil, true, false
	}
	if _, ok := openAIResponsesContextManagementAllowedTypes[typ]; !ok {
		return nil, true, false
	}
	if rawType, ok := normalized["type"]; !ok || rawType != typ {
		normalized["type"] = typ
		modified = true
	}
	if typ != "compaction" {
		if _, ok := normalized["compact_threshold"]; ok {
			delete(normalized, "compact_threshold")
			modified = true
		}
		return normalized, modified, true
	}
	if threshold, ok := intValueFromAny(normalized["compact_threshold"]); ok {
		if rawThreshold, exists := normalized["compact_threshold"]; !exists || rawThreshold != threshold {
			normalized["compact_threshold"] = threshold
			modified = true
		}
		return normalized, modified, true
	}
	delete(normalized, "compact_threshold")
	modified = true
	return normalized, modified, true
}

func intValueFromAny(value any) (int, bool) {
	switch v := value.(type) {
	case float64:
		return int(v), true
	case float32:
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
func sanitizeOpenAIResponsesReasoning(reasoning map[string]any) bool {
	modified := false
	for key := range reasoning {
		if _, ok := openAIResponsesReasoningAllowedKeys[key]; ok {
			continue
		}
		delete(reasoning, key)
		modified = true
	}

	effort := normalizeOpenAIResponsesReasoningEffort(stringValueFromAny(reasoning["effort"]))
	if rawEffort, ok := reasoning["effort"]; ok {
		if effort == "" {
			delete(reasoning, "effort")
			modified = true
		} else if rawEffort != effort {
			reasoning["effort"] = effort
			modified = true
		}
	}

	summary := normalizeOpenAIResponsesReasoningSummary(stringValueFromAny(reasoning["summary"]))
	if rawSummary, ok := reasoning["summary"]; ok {
		if summary == "" {
			delete(reasoning, "summary")
			modified = true
		} else if rawSummary != summary {
			reasoning["summary"] = summary
			modified = true
		}
	}

	return modified
}

func sanitizeOpenAIResponsesToolChoice(value any) (any, bool, bool) {
	if value == nil {
		return nil, false, false
	}

	switch v := value.(type) {
	case string:
		normalized := strings.TrimSpace(v)
		switch normalized {
		case "auto", "none", "required":
			if normalized == v {
				return value, false, true
			}
			return normalized, true, true
		default:
			return nil, true, false
		}
	case map[string]any:
		typ := strings.TrimSpace(stringValueFromAny(v["type"]))
		if typ != "function" {
			return nil, true, false
		}
		function, ok := v["function"].(map[string]any)
		if !ok {
			return nil, true, false
		}
		name := strings.TrimSpace(stringValueFromAny(function["name"]))
		if name == "" {
			return nil, true, false
		}
		normalized := map[string]any{
			"type":     "function",
			"function": map[string]any{"name": name},
		}
		encodedOld, _ := json.Marshal(v)
		encodedNew, _ := json.Marshal(normalized)
		if string(encodedOld) == string(encodedNew) {
			return value, false, true
		}
		return normalized, true, true
	default:
		return nil, true, false
	}
}

func normalizeOpenAIResponsesReasoningEffort(effort string) string {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "minimal", "none":
		return "none"
	case "low", "medium", "high", "xhigh":
		return strings.ToLower(strings.TrimSpace(effort))
	default:
		return ""
	}
}

func normalizeOpenAIResponsesReasoningSummary(summary string) string {
	switch strings.ToLower(strings.TrimSpace(summary)) {
	case "auto", "concise", "detailed":
		return strings.ToLower(strings.TrimSpace(summary))
	default:
		return ""
	}
}

func stringValueFromAny(value any) string {
	str, _ := value.(string)
	return str
}
