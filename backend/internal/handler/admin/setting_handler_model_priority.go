package admin

import (
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/handler/dto"
	"github.com/Wei-Shaw/sub2api/internal/service"
)

func openAIModelPriorityRulesToDTO(rules []service.OpenAIModelPriorityRule) []dto.OpenAIModelPriorityRule {
	if len(rules) == 0 {
		return []dto.OpenAIModelPriorityRule{}
	}
	converted := make([]dto.OpenAIModelPriorityRule, 0, len(rules))
	for _, rule := range rules {
		converted = append(converted, dto.OpenAIModelPriorityRule{
			Prefix:              rule.Prefix,
			PreferredAccountIDs: append([]int64(nil), rule.PreferredAccountIDs...),
			Enabled:             rule.Enabled,
		})
	}
	return converted
}

func openAIModelPriorityRulesToService(rules []dto.OpenAIModelPriorityRule) []service.OpenAIModelPriorityRule {
	if len(rules) == 0 {
		return []service.OpenAIModelPriorityRule{}
	}
	converted := make([]service.OpenAIModelPriorityRule, 0, len(rules))
	for _, rule := range rules {
		converted = append(converted, service.OpenAIModelPriorityRule{
			Prefix:              strings.TrimSpace(rule.Prefix),
			PreferredAccountIDs: append([]int64(nil), rule.PreferredAccountIDs...),
			Enabled:             rule.Enabled,
		})
	}
	return converted
}
