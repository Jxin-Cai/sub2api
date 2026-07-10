package service

import (
	"encoding/json"
	"strings"
)

func cloneOpenAIModelPriorityRules(rules []OpenAIModelPriorityRule) []OpenAIModelPriorityRule {
	if len(rules) == 0 {
		return []OpenAIModelPriorityRule{}
	}
	cloned := make([]OpenAIModelPriorityRule, 0, len(rules))
	for _, rule := range rules {
		ids := append([]int64(nil), rule.PreferredAccountIDs...)
		cloned = append(cloned, OpenAIModelPriorityRule{
			Prefix:              rule.Prefix,
			PreferredAccountIDs: ids,
			Enabled:             rule.Enabled,
		})
	}
	return cloned
}

func normalizeOpenAIModelPriorityRules(rules []OpenAIModelPriorityRule) []OpenAIModelPriorityRule {
	if len(rules) == 0 {
		return []OpenAIModelPriorityRule{}
	}
	normalized := make([]OpenAIModelPriorityRule, 0, len(rules))
	seenPrefixes := make(map[string]struct{}, len(rules))
	for _, rule := range rules {
		prefix := strings.ToLower(strings.TrimSpace(rule.Prefix))
		if prefix == "" {
			continue
		}
		if _, exists := seenPrefixes[prefix]; exists {
			continue
		}
		seenPrefixes[prefix] = struct{}{}

		ids := make([]int64, 0, len(rule.PreferredAccountIDs))
		seenIDs := make(map[int64]struct{}, len(rule.PreferredAccountIDs))
		for _, id := range rule.PreferredAccountIDs {
			if id <= 0 {
				continue
			}
			if _, exists := seenIDs[id]; exists {
				continue
			}
			seenIDs[id] = struct{}{}
			ids = append(ids, id)
		}
		if len(ids) == 0 {
			continue
		}
		normalized = append(normalized, OpenAIModelPriorityRule{
			Prefix:              prefix,
			PreferredAccountIDs: ids,
			Enabled:             rule.Enabled,
		})
	}
	return normalized
}

func ParseOpenAIModelPriorityRules(raw string) []OpenAIModelPriorityRule {
	if strings.TrimSpace(raw) == "" {
		return []OpenAIModelPriorityRule{}
	}
	var rules []OpenAIModelPriorityRule
	if err := json.Unmarshal([]byte(raw), &rules); err != nil {
		return []OpenAIModelPriorityRule{}
	}
	return normalizeOpenAIModelPriorityRules(rules)
}

func MarshalOpenAIModelPriorityRules(rules []OpenAIModelPriorityRule) string {
	normalized := normalizeOpenAIModelPriorityRules(rules)
	encoded, err := json.Marshal(normalized)
	if err != nil {
		return "[]"
	}
	return string(encoded)
}
