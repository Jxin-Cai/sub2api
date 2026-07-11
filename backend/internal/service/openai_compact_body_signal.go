package service

import (
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

const openAICompactOverrideKey = "openai_compact_override"

func SetOpenAICompactOverride(c *gin.Context) {
	if c == nil {
		return
	}
	c.Set(openAICompactOverrideKey, true)
}

// HasCompactionTriggerInInput detects the Codex remote compact v2 body signal:
// an input item with type "compaction_trigger". When the client sends this
// inside a normal POST /v1/responses (instead of POST /v1/responses/compact),
// the request must still be treated as a compact request — otherwise the
// upstream path, model mapping, and body normalization are all wrong, causing
// Codex to receive a non-compact response and fail with:
//
//	"remote compaction v2 expected exactly one compaction output item, got 0"
//
// The gateway handler promotes such requests by rewriting the URL path to the
// compact form before stream parsing, compact body normalization, and
// compact-capable account scheduling, so both inbound forms share one code path.
func HasCompactionTriggerInInput(body []byte) bool {
	if len(body) == 0 {
		return false
	}
	input := gjson.GetBytes(body, "input")
	if !input.IsArray() {
		return false
	}
	found := false
	input.ForEach(func(_, item gjson.Result) bool {
		if item.Get("type").String() == "compaction_trigger" {
			found = true
			return false
		}
		return true
	})
	return found
}

func DetectAnthropicCompactIntent(body []byte) bool {
	if len(body) == 0 {
		return false
	}
	if gjson.GetBytes(body, "context_management.edits").Array() != nil {
		for _, edit := range gjson.GetBytes(body, "context_management.edits").Array() {
			if strings.TrimSpace(edit.Get("type").String()) == "compact_20260112" {
				return true
			}
		}
	}
	for _, msg := range gjson.GetBytes(body, "messages").Array() {
		content := msg.Get("content")
		if detectAnthropicCompactBlock(content) {
			return true
		}
	}
	return false
}

func detectAnthropicCompactBlock(content gjson.Result) bool {
	if content.IsArray() {
		found := false
		content.ForEach(func(_, block gjson.Result) bool {
			if strings.TrimSpace(block.Get("type").String()) == "compact_20260112" {
				found = true
				return false
			}
			return true
		})
		return found
	}
	return content.IsObject() && strings.TrimSpace(content.Get("type").String()) == "compact_20260112"
}
