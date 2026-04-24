package service

import (
	"encoding/json"
	"strings"

	"github.com/tidwall/gjson"
)

// contentSessionSeedPrefix prevents collisions between content-derived seeds
// and explicit session IDs (e.g. "sess-xxx" or "compat_cc_xxx").
const (
	contentSessionSeedPrefix   = "compat_cs_"
	reasoningSessionSeedPrefix = "compat_reasoning_"
)

func deriveOpenAIReasoningSessionSeed(body []byte) string {
	if len(body) == 0 {
		return ""
	}

	var signatures []string
	messages := gjson.GetBytes(body, "messages")
	if messages.Exists() && messages.IsArray() {
		messages.ForEach(func(_, msg gjson.Result) bool {
			content := msg.Get("content")
			if !content.Exists() || !content.IsArray() {
				return true
			}
			content.ForEach(func(_, block gjson.Result) bool {
				if block.Get("type").String() == "thinking" {
					if sig := strings.TrimSpace(block.Get("signature").String()); sig != "" {
						signatures = append(signatures, sig)
					}
				}
				return true
			})
			return true
		})
	}

	input := gjson.GetBytes(body, "input")
	if input.Exists() && input.IsArray() {
		input.ForEach(func(_, item gjson.Result) bool {
			itemType := strings.TrimSpace(item.Get("type").String())
			if itemType != "reasoning" && itemType != "compaction" {
				return true
			}
			id := strings.TrimSpace(item.Get("id").String())
			enc := strings.TrimSpace(item.Get("encrypted_content").String())
			if id != "" || enc != "" {
				signatures = append(signatures, itemType+"|"+id+"|"+enc)
			}
			return true
		})
	}

	if len(signatures) == 0 {
		return ""
	}
	return reasoningSessionSeedPrefix + hashSensitiveValueForLog(strings.Join(signatures, "|"))
}

// deriveOpenAIContentSessionSeed builds a stable session seed from an
// OpenAI-format request body. Only fields constant across conversation turns
// are included: model, tools/functions definitions, system/developer prompts,
// instructions (Responses API), and the first user message.
// Supports both Chat Completions (messages) and Responses API (input).
func deriveOpenAIContentSessionSeed(body []byte) string {
	if len(body) == 0 {
		return ""
	}

	var b strings.Builder

	if model := gjson.GetBytes(body, "model").String(); model != "" {
		_, _ = b.WriteString("model=")
		_, _ = b.WriteString(model)
	}

	if tools := gjson.GetBytes(body, "tools"); tools.Exists() && tools.IsArray() && tools.Raw != "[]" {
		_, _ = b.WriteString("|tools=")
		_, _ = b.WriteString(normalizeCompatSeedJSON(json.RawMessage(tools.Raw)))
	}

	if funcs := gjson.GetBytes(body, "functions"); funcs.Exists() && funcs.IsArray() && funcs.Raw != "[]" {
		_, _ = b.WriteString("|functions=")
		_, _ = b.WriteString(normalizeCompatSeedJSON(json.RawMessage(funcs.Raw)))
	}

	if instr := gjson.GetBytes(body, "instructions").String(); instr != "" {
		_, _ = b.WriteString("|instructions=")
		_, _ = b.WriteString(instr)
	}

	firstUserCaptured := false

	msgs := gjson.GetBytes(body, "messages")
	if msgs.Exists() && msgs.IsArray() {
		msgs.ForEach(func(_, msg gjson.Result) bool {
			role := msg.Get("role").String()
			switch role {
			case "system", "developer":
				_, _ = b.WriteString("|system=")
				if c := msg.Get("content"); c.Exists() {
					_, _ = b.WriteString(normalizeCompatSeedJSON(json.RawMessage(c.Raw)))
				}
			case "user":
				if !firstUserCaptured {
					_, _ = b.WriteString("|first_user=")
					if c := msg.Get("content"); c.Exists() {
						_, _ = b.WriteString(normalizeCompatSeedJSON(json.RawMessage(c.Raw)))
					}
					firstUserCaptured = true
				}
			}
			return true
		})
	} else if inp := gjson.GetBytes(body, "input"); inp.Exists() {
		if inp.Type == gjson.String {
			_, _ = b.WriteString("|input=")
			_, _ = b.WriteString(inp.String())
		} else if inp.IsArray() {
			inp.ForEach(func(_, item gjson.Result) bool {
				role := item.Get("role").String()
				switch role {
				case "system", "developer":
					_, _ = b.WriteString("|system=")
					if c := item.Get("content"); c.Exists() {
						_, _ = b.WriteString(normalizeCompatSeedJSON(json.RawMessage(c.Raw)))
					}
				case "user":
					if !firstUserCaptured {
						_, _ = b.WriteString("|first_user=")
						if c := item.Get("content"); c.Exists() {
							_, _ = b.WriteString(normalizeCompatSeedJSON(json.RawMessage(c.Raw)))
						}
						firstUserCaptured = true
					}
				}
				if !firstUserCaptured && item.Get("type").String() == "input_text" {
					_, _ = b.WriteString("|first_user=")
					if text := item.Get("text").String(); text != "" {
						_, _ = b.WriteString(text)
					}
					firstUserCaptured = true
				}
				return true
			})
		}
	}

	if b.Len() == 0 {
		return ""
	}
	return contentSessionSeedPrefix + b.String()
}
