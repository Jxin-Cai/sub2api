package admin

import (
	"context"
	"sort"

	"github.com/Wei-Shaw/sub2api/internal/pkg/response"
	"github.com/gin-gonic/gin"
)

type DedupDuplicateGroup struct {
	Name      string  `json:"name"`
	KeepID    int64   `json:"keep_id"`
	RemoveIDs []int64 `json:"remove_ids"`
	Count     int     `json:"count"`
}

type DedupPreviewResult struct {
	Duplicates  []DedupDuplicateGroup `json:"duplicates"`
	TotalRemove int                   `json:"total_remove"`
}

type DedupExecuteResult struct {
	Removed int `json:"removed"`
}

func (h *AccountHandler) DedupPreview(c *gin.Context) {
	ctx := c.Request.Context()

	result, err := h.findDuplicateAccounts(ctx)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}

	response.Success(c, result)
}

func (h *AccountHandler) DedupExecute(c *gin.Context) {
	ctx := c.Request.Context()

	preview, err := h.findDuplicateAccounts(ctx)
	if err != nil {
		response.ErrorFrom(c, err)
		return
	}

	removed := 0
	for _, group := range preview.Duplicates {
		for _, id := range group.RemoveIDs {
			if delErr := h.adminService.DeleteAccount(ctx, id); delErr != nil {
				continue
			}
			removed++
		}
	}

	response.Success(c, DedupExecuteResult{Removed: removed})
}

func (h *AccountHandler) findDuplicateAccounts(ctx context.Context) (*DedupPreviewResult, error) {
	accounts, err := h.listAccountsFiltered(ctx, "", "", "", "", 0, "", "created_at", "asc")
	if err != nil {
		return nil, err
	}

	type accountEntry struct {
		ID        int64
		CreatedAt int64
	}

	nameGroups := make(map[string][]accountEntry)
	for _, acc := range accounts {
		createdAt := acc.CreatedAt.Unix()
		nameGroups[acc.Name] = append(nameGroups[acc.Name], accountEntry{
			ID:        acc.ID,
			CreatedAt: createdAt,
		})
	}

	var duplicates []DedupDuplicateGroup
	totalRemove := 0

	for name, entries := range nameGroups {
		if len(entries) <= 1 {
			continue
		}
		sort.Slice(entries, func(i, j int) bool {
			return entries[i].CreatedAt < entries[j].CreatedAt
		})

		keepID := entries[0].ID
		removeIDs := make([]int64, 0, len(entries)-1)
		for _, e := range entries[1:] {
			removeIDs = append(removeIDs, e.ID)
		}

		duplicates = append(duplicates, DedupDuplicateGroup{
			Name:      name,
			KeepID:    keepID,
			RemoveIDs: removeIDs,
			Count:     len(entries),
		})
		totalRemove += len(removeIDs)
	}

	sort.Slice(duplicates, func(i, j int) bool {
		return duplicates[i].Count > duplicates[j].Count
	})

	return &DedupPreviewResult{
		Duplicates:  duplicates,
		TotalRemove: totalRemove,
	}, nil
}
