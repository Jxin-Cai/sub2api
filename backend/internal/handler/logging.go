package handler

import (
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

func requestLogger(c *gin.Context, component string, fields ...zap.Field) *zap.Logger {
	base := logger.L()
	if c != nil && c.Request != nil {
		base = logger.FromContext(c.Request.Context())
	}

	if component != "" {
		fields = append([]zap.Field{zap.String("component", component)}, fields...)
	}
	return base.With(fields...)
}

type requestCompletedLogInput struct {
	Endpoint             string
	Model                string
	UpstreamModel        string
	AccountID            int64
	StatusCode           int
	InputTokens          int
	CacheReadInputTokens int
	OutputTokens         int
	Stream               bool
	Duration             time.Duration
	FirstTokenMs         *int
}

func upstreamErrorLogFields(c *gin.Context) []zap.Field {
	if c == nil {
		return nil
	}
	fields := make([]zap.Field, 0, 6)
	if v, ok := c.Get(service.OpsUpstreamStatusCodeKey); ok {
		switch t := v.(type) {
		case int:
			if t > 0 {
				fields = append(fields, zap.Int("upstream_status_code", t))
			}
		case int64:
			if t > 0 {
				fields = append(fields, zap.Int64("upstream_status_code", t))
			}
		}
	}
	if v, ok := c.Get(service.OpsUpstreamErrorMessageKey); ok {
		if msg, ok := v.(string); ok && strings.TrimSpace(msg) != "" {
			fields = append(fields, zap.String("upstream_error_message", msg))
		}
	}
	if v, ok := c.Get(service.OpsUpstreamErrorDetailKey); ok {
		if detail, ok := v.(string); ok && strings.TrimSpace(detail) != "" {
			fields = append(fields, zap.String("upstream_error_detail", detail))
		}
	}
	if v, ok := c.Get(service.OpsUpstreamErrorsKey); ok {
		if events, ok := v.([]*service.OpsUpstreamErrorEvent); ok && len(events) > 0 {
			fields = append(fields, zap.Int("upstream_errors_count", len(events)), zap.Any("upstream_errors", events))
			last := events[len(events)-1]
			if last != nil {
				if last.UpstreamStatusCode > 0 {
					fields = append(fields, zap.Int("last_upstream_status_code", last.UpstreamStatusCode))
				}
				if strings.TrimSpace(last.Kind) != "" {
					fields = append(fields, zap.String("last_upstream_error_kind", last.Kind))
				}
				if strings.TrimSpace(last.Message) != "" {
					fields = append(fields, zap.String("last_upstream_error_message", last.Message))
				}
				if strings.TrimSpace(last.UpstreamRequestID) != "" {
					fields = append(fields, zap.String("last_upstream_request_id", last.UpstreamRequestID))
				}
			}
		}
	}
	return fields
}

func logRequestCompletedCompact(reqLog *zap.Logger, event string, input requestCompletedLogInput, extra ...zap.Field) {
	if reqLog == nil {
		reqLog = logger.L()
	}
	statusCode := input.StatusCode
	if statusCode == 0 {
		statusCode = 200
	}
	fields := []zap.Field{
		zap.String("endpoint", input.Endpoint),
		zap.String("model", input.Model),
		zap.Int64("account_id", input.AccountID),
		zap.Int("status_code", statusCode),
		zap.Int("input_tokens", input.InputTokens),
		zap.Int("cache_read_input_tokens", input.CacheReadInputTokens),
		zap.Int("output_tokens", input.OutputTokens),
		zap.Bool("stream", input.Stream),
	}
	if input.UpstreamModel != "" && input.UpstreamModel != input.Model {
		fields = append(fields, zap.String("upstream_model", input.UpstreamModel))
	}
	if input.Duration > 0 {
		fields = append(fields, zap.Int64("duration_ms", input.Duration.Milliseconds()))
	}
	if input.FirstTokenMs != nil {
		fields = append(fields, zap.Intp("first_token_ms", input.FirstTokenMs))
	}
	fields = append(fields, extra...)
	reqLog.Info(event, fields...)
}
