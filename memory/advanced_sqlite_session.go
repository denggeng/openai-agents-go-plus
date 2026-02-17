// Copyright 2025 The NLP Odyssey Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package memory

import (
	"cmp"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/denggeng/openai-agents-go-plus/usage"
	"github.com/openai/openai-go/v3/responses"
)

const MainBranchID = "main"

// AdvancedSQLiteSession adds branch-aware conversation history management
// on top of SQLiteSession.
type AdvancedSQLiteSession struct {
	rootSessionID  string
	dbDSN          string
	sessionTable   string
	messagesTable  string
	turnUsageTable string

	currentBranchID string
	branches        map[string]*SQLiteSession

	mu sync.Mutex
}

type AdvancedSQLiteSessionParams struct {
	SessionID string

	// Optional database data source name. Defaults to `file::memory:?cache=shared`.
	DBDataSourceName string

	// Optional table names (shared by all branches).
	SessionTable   string
	MessagesTable  string
	TurnUsageTable string
}

type AdvancedConversationTurn struct {
	Turn        int    `json:"turn"`
	Content     string `json:"content"`
	FullContent string `json:"full_content"`
	Timestamp   string `json:"timestamp"`
	CanBranch   bool   `json:"can_branch"`
}

type AdvancedConversationTurnItem struct {
	Type     string `json:"type"`
	ToolName string `json:"tool_name,omitempty"`
}

type AdvancedToolUsage struct {
	ToolName   string `json:"tool_name"`
	UsageCount int    `json:"usage_count"`
	Turn       int    `json:"turn"`
}

type AdvancedBranchInfo struct {
	BranchID     string `json:"branch_id"`
	MessageCount int    `json:"message_count"`
	UserTurns    int    `json:"user_turns"`
	IsCurrent    bool   `json:"is_current"`
	CreatedAt    string `json:"created_at"`
}

type AdvancedSessionUsage struct {
	Requests     uint64 `json:"requests"`
	InputTokens  uint64 `json:"input_tokens"`
	OutputTokens uint64 `json:"output_tokens"`
	TotalTokens  uint64 `json:"total_tokens"`
	TotalTurns   int    `json:"total_turns"`
}

type AdvancedTurnUsage struct {
	UserTurnNumber      int                                        `json:"user_turn_number"`
	Requests            uint64                                     `json:"requests"`
	InputTokens         uint64                                     `json:"input_tokens"`
	OutputTokens        uint64                                     `json:"output_tokens"`
	TotalTokens         uint64                                     `json:"total_tokens"`
	InputTokensDetails  responses.ResponseUsageInputTokensDetails  `json:"input_tokens_details"`
	OutputTokensDetails responses.ResponseUsageOutputTokensDetails `json:"output_tokens_details"`
}

func NewAdvancedSQLiteSession(ctx context.Context, params AdvancedSQLiteSessionParams) (*AdvancedSQLiteSession, error) {
	if strings.TrimSpace(params.SessionID) == "" {
		return nil, fmt.Errorf("session id is required")
	}

	s := &AdvancedSQLiteSession{
		rootSessionID:   params.SessionID,
		dbDSN:           cmp.Or(params.DBDataSourceName, "file::memory:?cache=shared"),
		sessionTable:    cmp.Or(params.SessionTable, "agent_sessions"),
		messagesTable:   cmp.Or(params.MessagesTable, "agent_messages"),
		turnUsageTable:  cmp.Or(params.TurnUsageTable, "turn_usage"),
		currentBranchID: MainBranchID,
		branches:        make(map[string]*SQLiteSession),
	}

	mainSession, err := s.newBranchSession(ctx, MainBranchID)
	if err != nil {
		return nil, err
	}
	s.branches[MainBranchID] = mainSession

	if err := s.initTurnUsageTable(ctx, mainSession.db); err != nil {
		_ = mainSession.Close()
		return nil, err
	}

	return s, nil
}

func (s *AdvancedSQLiteSession) SessionID(context.Context) string {
	return s.rootSessionID
}

func (s *AdvancedSQLiteSession) CurrentBranchID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.currentBranchID
}

func (s *AdvancedSQLiteSession) ListBranches() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	branches := make([]string, 0, len(s.branches))
	for branchID := range s.branches {
		branches = append(branches, branchID)
	}
	slices.Sort(branches)
	return branches
}

// ListBranchInfos returns branch metadata similar to Python's list_branches output.
func (s *AdvancedSQLiteSession) ListBranchInfos(ctx context.Context) ([]AdvancedBranchInfo, error) {
	s.mu.Lock()
	branchIDs := make([]string, 0, len(s.branches))
	branchSessions := make(map[string]*SQLiteSession, len(s.branches))
	currentBranchID := s.currentBranchID
	for branchID, branchSession := range s.branches {
		branchIDs = append(branchIDs, branchID)
		branchSessions[branchID] = branchSession
	}
	s.mu.Unlock()
	slices.Sort(branchIDs)

	infos := make([]AdvancedBranchInfo, 0, len(branchIDs))
	for _, branchID := range branchIDs {
		branchSession := branchSessions[branchID]
		if branchSession == nil {
			continue
		}

		rows, err := branchSession.db.QueryContext(
			ctx,
			fmt.Sprintf(`
				SELECT message_data, created_at
				FROM "%s"
				WHERE session_id = ?
				ORDER BY created_at ASC
			`, s.messagesTable),
			branchSession.sessionID,
		)
		if err != nil {
			return nil, err
		}

		messageCount := 0
		userTurns := 0
		createdAt := ""
		for rows.Next() {
			var messageData string
			var createdAtRaw any
			if err := rows.Scan(&messageData, &createdAtRaw); err != nil {
				_ = rows.Close()
				return nil, err
			}

			messageCount++
			if createdAt == "" {
				createdAt = dbTimestampToString(createdAtRaw)
			}

			item, err := unmarshalMessageData(messageData)
			if err != nil {
				continue
			}
			if isUserMessageItem(item) {
				userTurns++
			}
		}
		if err := rows.Err(); err != nil {
			_ = rows.Close()
			return nil, err
		}
		_ = rows.Close()

		infos = append(infos, AdvancedBranchInfo{
			BranchID:     branchID,
			MessageCount: messageCount,
			UserTurns:    userTurns,
			IsCurrent:    branchID == currentBranchID,
			CreatedAt:    createdAt,
		})
	}

	return infos, nil
}

func (s *AdvancedSQLiteSession) SwitchToBranch(branchID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.branches[branchID]; !ok {
		return fmt.Errorf("branch %q does not exist", branchID)
	}
	s.currentBranchID = branchID
	return nil
}

func (s *AdvancedSQLiteSession) CreateBranchFromTurn(ctx context.Context, turn int, branchID string) (string, error) {
	if turn <= 0 {
		return "", fmt.Errorf("turn must be > 0")
	}
	branchID = strings.TrimSpace(branchID)
	if branchID == "" {
		return "", fmt.Errorf("branch id is required")
	}

	s.mu.Lock()
	if _, exists := s.branches[branchID]; exists {
		s.mu.Unlock()
		return "", fmt.Errorf("branch %q already exists", branchID)
	}
	currentSession, ok := s.branches[s.currentBranchID]
	s.mu.Unlock()
	if !ok {
		return "", fmt.Errorf("current branch %q is missing", s.currentBranchID)
	}

	items, err := currentSession.GetItems(ctx, 0)
	if err != nil {
		return "", err
	}

	copiedItems := itemsBeforeTurn(items, turn)

	newBranchSession, err := s.newBranchSession(ctx, branchID)
	if err != nil {
		return "", err
	}

	if err := newBranchSession.ClearSession(ctx); err != nil {
		_ = newBranchSession.Close()
		return "", err
	}
	if len(copiedItems) > 0 {
		if err := newBranchSession.AddItems(ctx, copiedItems); err != nil {
			_ = newBranchSession.Close()
			return "", err
		}
	}

	s.mu.Lock()
	s.branches[branchID] = newBranchSession
	s.currentBranchID = branchID
	s.mu.Unlock()
	return branchID, nil
}

// CreateBranchFromContent creates a new branch from the first user turn
// containing searchTerm. When branchID is empty, a timestamped name is generated.
func (s *AdvancedSQLiteSession) CreateBranchFromContent(ctx context.Context, searchTerm, branchID string) (string, error) {
	matches, err := s.FindTurnsByContent(ctx, searchTerm, "")
	if err != nil {
		return "", err
	}
	if len(matches) == 0 {
		return "", fmt.Errorf("no user turns found containing %q", searchTerm)
	}

	branchID = strings.TrimSpace(branchID)
	if branchID == "" {
		branchID = fmt.Sprintf("branch_from_turn_%d_%d", matches[0].Turn, time.Now().Unix())
	}

	return s.CreateBranchFromTurn(ctx, matches[0].Turn, branchID)
}

func (s *AdvancedSQLiteSession) DeleteBranch(ctx context.Context, branchID string) error {
	return s.deleteBranch(ctx, branchID, false)
}

// DeleteBranchForce deletes branchID even if it is currently selected.
// When deleting the active branch, the current branch is switched to `main`.
func (s *AdvancedSQLiteSession) DeleteBranchForce(ctx context.Context, branchID string) error {
	return s.deleteBranch(ctx, branchID, true)
}

func (s *AdvancedSQLiteSession) deleteBranch(ctx context.Context, branchID string, force bool) error {
	branchID = strings.TrimSpace(branchID)
	if branchID == "" {
		return fmt.Errorf("branch id cannot be empty")
	}
	if branchID == MainBranchID {
		return fmt.Errorf("cannot delete main branch")
	}

	s.mu.Lock()
	if branchID == s.currentBranchID {
		if !force {
			s.mu.Unlock()
			return fmt.Errorf("cannot delete current branch %q without force", branchID)
		}
		s.currentBranchID = MainBranchID
	}

	branchSession, ok := s.branches[branchID]
	if !ok {
		s.mu.Unlock()
		return fmt.Errorf("branch %q does not exist", branchID)
	}
	delete(s.branches, branchID)
	s.mu.Unlock()

	if err := branchSession.ClearSession(ctx); err != nil {
		_ = branchSession.Close()
		return err
	}
	if _, err := branchSession.db.ExecContext(
		ctx,
		fmt.Sprintf(`DELETE FROM "%s" WHERE session_id = ? AND branch_id = ?`, s.turnUsageTable),
		s.rootSessionID,
		branchID,
	); err != nil {
		_ = branchSession.Close()
		return err
	}
	return branchSession.Close()
}

// GetConversationTurns returns the user turns in order for the selected branch.
// If branchID is empty, the current branch is used.
func (s *AdvancedSQLiteSession) GetConversationTurns(ctx context.Context, branchID string) ([]AdvancedConversationTurn, error) {
	items, err := s.getBranchItemsWithCreatedAt(ctx, branchID)
	if err != nil {
		return nil, err
	}

	turns := make([]AdvancedConversationTurn, 0)
	userTurn := 0
	for _, item := range items {
		content, ok := extractUserMessageContent(item.Item)
		if !ok {
			continue
		}
		userTurn++
		turns = append(turns, AdvancedConversationTurn{
			Turn:        userTurn,
			Content:     truncateWithEllipsis(content, 100),
			FullContent: content,
			Timestamp:   item.CreatedAt,
			CanBranch:   true,
		})
	}
	return turns, nil
}

// FindTurnsByContent returns user turns containing searchTerm for the selected branch.
// If branchID is empty, the current branch is used.
func (s *AdvancedSQLiteSession) FindTurnsByContent(ctx context.Context, searchTerm, branchID string) ([]AdvancedConversationTurn, error) {
	items, err := s.getBranchItemsWithCreatedAt(ctx, branchID)
	if err != nil {
		return nil, err
	}

	searchTermLower := strings.ToLower(searchTerm)
	matches := make([]AdvancedConversationTurn, 0)
	userTurn := 0
	for _, item := range items {
		content, ok := extractUserMessageContent(item.Item)
		if !ok {
			continue
		}

		userTurn++
		if searchTermLower != "" && !strings.Contains(strings.ToLower(content), searchTermLower) {
			continue
		}
		matches = append(matches, AdvancedConversationTurn{
			Turn:        userTurn,
			Content:     content,
			FullContent: content,
			Timestamp:   item.CreatedAt,
			CanBranch:   true,
		})
	}
	return matches, nil
}

// GetConversationByTurns groups item metadata by user turn number for the selected branch.
// If branchID is empty, the current branch is used.
func (s *AdvancedSQLiteSession) GetConversationByTurns(
	ctx context.Context, branchID string,
) (map[int][]AdvancedConversationTurnItem, error) {
	items, err := s.getBranchItems(ctx, branchID)
	if err != nil {
		return nil, err
	}

	turns := make(map[int][]AdvancedConversationTurnItem)
	userTurn := 0
	for _, item := range items {
		if isUserMessageItem(item) {
			userTurn++
		}

		itemType := inputItemType(item)
		turns[userTurn] = append(turns[userTurn], AdvancedConversationTurnItem{
			Type:     itemType,
			ToolName: extractToolName(item, itemType),
		})
	}
	return turns, nil
}

// GetToolUsage returns tool usage count grouped by tool name and user turn.
// If branchID is empty, the current branch is used.
func (s *AdvancedSQLiteSession) GetToolUsage(ctx context.Context, branchID string) ([]AdvancedToolUsage, error) {
	items, err := s.getBranchItems(ctx, branchID)
	if err != nil {
		return nil, err
	}

	type toolUsageKey struct {
		toolName string
		userTurn int
	}

	counts := make(map[toolUsageKey]int)
	userTurn := 0
	for _, item := range items {
		if isUserMessageItem(item) {
			userTurn++
		}

		itemType := inputItemType(item)
		if !isToolCallItemType(itemType) {
			continue
		}

		toolName := extractToolName(item, itemType)
		if toolName == "" {
			toolName = itemType
		}

		counts[toolUsageKey{toolName: toolName, userTurn: userTurn}]++
	}

	usage := make([]AdvancedToolUsage, 0, len(counts))
	for key, count := range counts {
		usage = append(usage, AdvancedToolUsage{
			ToolName:   key.toolName,
			UsageCount: count,
			Turn:       key.userTurn,
		})
	}
	slices.SortFunc(usage, func(a, b AdvancedToolUsage) int {
		if c := cmp.Compare(a.Turn, b.Turn); c != 0 {
			return c
		}
		return cmp.Compare(a.ToolName, b.ToolName)
	})
	return usage, nil
}

// StoreRunUsage aggregates usage data for the current user turn in the active branch.
func (s *AdvancedSQLiteSession) StoreRunUsage(ctx context.Context, runUsage *usage.Usage) error {
	if runUsage == nil {
		return nil
	}

	branchID := s.CurrentBranchID()
	userTurn, err := s.currentUserTurnNumber(ctx, branchID)
	if err != nil {
		return err
	}
	if userTurn <= 0 {
		return nil
	}

	return s.upsertTurnUsage(ctx, branchID, userTurn, runUsage)
}

// GetSessionUsage returns cumulative usage for all branches or one specific branch.
// If branchID is empty, usage is aggregated across all branches.
func (s *AdvancedSQLiteSession) GetSessionUsage(ctx context.Context, branchID string) (*AdvancedSessionUsage, error) {
	s.mu.Lock()
	mainSession, ok := s.branches[MainBranchID]
	s.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("main branch %q is missing", MainBranchID)
	}

	branchID = strings.TrimSpace(branchID)
	var row *sql.Row
	if branchID == "" {
		row = mainSession.db.QueryRowContext(
			ctx,
			fmt.Sprintf(`
				SELECT
					SUM(requests),
					SUM(input_tokens),
					SUM(output_tokens),
					SUM(total_tokens),
					COUNT(*)
				FROM "%s"
				WHERE session_id = ?
			`, s.turnUsageTable),
			s.rootSessionID,
		)
	} else {
		row = mainSession.db.QueryRowContext(
			ctx,
			fmt.Sprintf(`
				SELECT
					SUM(requests),
					SUM(input_tokens),
					SUM(output_tokens),
					SUM(total_tokens),
					COUNT(*)
				FROM "%s"
				WHERE session_id = ? AND branch_id = ?
			`, s.turnUsageTable),
			s.rootSessionID,
			branchID,
		)
	}

	var requests sql.NullInt64
	var inputTokens sql.NullInt64
	var outputTokens sql.NullInt64
	var totalTokens sql.NullInt64
	var totalTurns int
	if err := row.Scan(&requests, &inputTokens, &outputTokens, &totalTokens, &totalTurns); err != nil {
		return nil, err
	}
	if !requests.Valid {
		return nil, nil
	}

	return &AdvancedSessionUsage{
		Requests:     uint64(maxInt64ToZero(requests.Int64)),
		InputTokens:  uint64(maxInt64ToZero(inputTokens.Int64)),
		OutputTokens: uint64(maxInt64ToZero(outputTokens.Int64)),
		TotalTokens:  uint64(maxInt64ToZero(totalTokens.Int64)),
		TotalTurns:   totalTurns,
	}, nil
}

// GetTurnUsage returns usage for the selected branch.
// If userTurnNumber is omitted, all turns are returned in ascending order.
// If branchID is empty, the current branch is used.
func (s *AdvancedSQLiteSession) GetTurnUsage(
	ctx context.Context, branchID string, userTurnNumber ...int,
) ([]AdvancedTurnUsage, error) {
	if len(userTurnNumber) > 1 {
		return nil, fmt.Errorf("expected at most one userTurnNumber")
	}

	branchID, err := s.resolveBranchID(branchID)
	if err != nil {
		return nil, err
	}

	branchSession, err := s.branchSession(branchID)
	if err != nil {
		return nil, err
	}

	var rows *sql.Rows
	if len(userTurnNumber) == 1 {
		rows, err = branchSession.db.QueryContext(
			ctx,
			fmt.Sprintf(`
				SELECT
					user_turn_number,
					requests,
					input_tokens,
					output_tokens,
					total_tokens,
					input_tokens_details,
					output_tokens_details
				FROM "%s"
				WHERE session_id = ? AND branch_id = ? AND user_turn_number = ?
			`, s.turnUsageTable),
			s.rootSessionID,
			branchID,
			userTurnNumber[0],
		)
	} else {
		rows, err = branchSession.db.QueryContext(
			ctx,
			fmt.Sprintf(`
				SELECT
					user_turn_number,
					requests,
					input_tokens,
					output_tokens,
					total_tokens,
					input_tokens_details,
					output_tokens_details
				FROM "%s"
				WHERE session_id = ? AND branch_id = ?
				ORDER BY user_turn_number
			`, s.turnUsageTable),
			s.rootSessionID,
			branchID,
		)
	}
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = rows.Close()
	}()

	turnUsage := make([]AdvancedTurnUsage, 0)
	for rows.Next() {
		var userTurn int
		var requests int64
		var inputTokens int64
		var outputTokens int64
		var totalTokens int64
		var inputDetailsRaw sql.NullString
		var outputDetailsRaw sql.NullString
		if err := rows.Scan(
			&userTurn,
			&requests,
			&inputTokens,
			&outputTokens,
			&totalTokens,
			&inputDetailsRaw,
			&outputDetailsRaw,
		); err != nil {
			return nil, err
		}

		inputDetails, outputDetails := decodeUsageDetails(inputDetailsRaw, outputDetailsRaw)
		turnUsage = append(turnUsage, AdvancedTurnUsage{
			UserTurnNumber:      userTurn,
			Requests:            uint64(maxInt64ToZero(requests)),
			InputTokens:         uint64(maxInt64ToZero(inputTokens)),
			OutputTokens:        uint64(maxInt64ToZero(outputTokens)),
			TotalTokens:         uint64(maxInt64ToZero(totalTokens)),
			InputTokensDetails:  inputDetails,
			OutputTokensDetails: outputDetails,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	return turnUsage, nil
}

func (s *AdvancedSQLiteSession) GetItems(ctx context.Context, limit int) ([]TResponseInputItem, error) {
	branchSession, err := s.currentBranchSession()
	if err != nil {
		return nil, err
	}
	return branchSession.GetItems(ctx, limit)
}

func (s *AdvancedSQLiteSession) AddItems(ctx context.Context, items []TResponseInputItem) error {
	branchSession, err := s.currentBranchSession()
	if err != nil {
		return err
	}
	return branchSession.AddItems(ctx, items)
}

func (s *AdvancedSQLiteSession) PopItem(ctx context.Context) (*TResponseInputItem, error) {
	branchSession, err := s.currentBranchSession()
	if err != nil {
		return nil, err
	}
	return branchSession.PopItem(ctx)
}

func (s *AdvancedSQLiteSession) ClearSession(ctx context.Context) error {
	branchSession, err := s.currentBranchSession()
	if err != nil {
		return err
	}
	return branchSession.ClearSession(ctx)
}

func (s *AdvancedSQLiteSession) Close() error {
	s.mu.Lock()
	branchSessions := make([]*SQLiteSession, 0, len(s.branches))
	for _, branchSession := range s.branches {
		branchSessions = append(branchSessions, branchSession)
	}
	s.mu.Unlock()

	var closeErr error
	for _, branchSession := range branchSessions {
		if err := branchSession.Close(); err != nil {
			closeErr = err
		}
	}
	return closeErr
}

func (s *AdvancedSQLiteSession) currentBranchSession() (*SQLiteSession, error) {
	return s.branchSession("")
}

func (s *AdvancedSQLiteSession) branchSession(branchID string) (*SQLiteSession, error) {
	branchID = strings.TrimSpace(branchID)

	s.mu.Lock()
	defer s.mu.Unlock()
	if branchID == "" {
		branchID = s.currentBranchID
	}

	branchSession, ok := s.branches[branchID]
	if !ok {
		return nil, fmt.Errorf("branch %q does not exist", branchID)
	}
	return branchSession, nil
}

func (s *AdvancedSQLiteSession) getBranchItems(
	ctx context.Context, branchID string,
) ([]TResponseInputItem, error) {
	branchSession, err := s.branchSession(branchID)
	if err != nil {
		return nil, err
	}
	return branchSession.GetItems(ctx, 0)
}

type advancedTimedInputItem struct {
	Item      TResponseInputItem
	CreatedAt string
}

func (s *AdvancedSQLiteSession) getBranchItemsWithCreatedAt(
	ctx context.Context, branchID string,
) ([]advancedTimedInputItem, error) {
	branchSession, err := s.branchSession(branchID)
	if err != nil {
		return nil, err
	}

	rows, err := branchSession.db.QueryContext(
		ctx,
		fmt.Sprintf(`
			SELECT message_data, created_at
			FROM "%s"
			WHERE session_id = ?
			ORDER BY created_at ASC
		`, s.messagesTable),
		branchSession.sessionID,
	)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = rows.Close()
	}()

	items := make([]advancedTimedInputItem, 0)
	for rows.Next() {
		var messageData string
		var createdAtRaw any
		if err := rows.Scan(&messageData, &createdAtRaw); err != nil {
			return nil, err
		}

		item, err := unmarshalMessageData(messageData)
		if err != nil {
			continue
		}

		items = append(items, advancedTimedInputItem{
			Item:      item,
			CreatedAt: dbTimestampToString(createdAtRaw),
		})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	return items, nil
}

func (s *AdvancedSQLiteSession) resolveBranchID(branchID string) (string, error) {
	branchID = strings.TrimSpace(branchID)
	s.mu.Lock()
	defer s.mu.Unlock()
	if branchID == "" {
		branchID = s.currentBranchID
	}
	if _, ok := s.branches[branchID]; !ok {
		return "", fmt.Errorf("branch %q does not exist", branchID)
	}
	return branchID, nil
}

func (s *AdvancedSQLiteSession) currentUserTurnNumber(ctx context.Context, branchID string) (int, error) {
	items, err := s.getBranchItems(ctx, branchID)
	if err != nil {
		return 0, err
	}

	userTurn := 0
	for _, item := range items {
		if isUserMessageItem(item) {
			userTurn++
		}
	}
	return userTurn, nil
}

func (s *AdvancedSQLiteSession) upsertTurnUsage(
	ctx context.Context,
	branchID string,
	userTurn int,
	runUsage *usage.Usage,
) error {
	branchSession, err := s.branchSession(branchID)
	if err != nil {
		return err
	}

	inputDetailsRaw, err := json.Marshal(runUsage.InputTokensDetails)
	if err != nil {
		return err
	}
	outputDetailsRaw, err := json.Marshal(runUsage.OutputTokensDetails)
	if err != nil {
		return err
	}

	_, err = branchSession.db.ExecContext(
		ctx,
		fmt.Sprintf(`
			INSERT INTO "%s" (
				session_id,
				branch_id,
				user_turn_number,
				requests,
				input_tokens,
				output_tokens,
				total_tokens,
				input_tokens_details,
				output_tokens_details
			)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(session_id, branch_id, user_turn_number) DO UPDATE SET
				requests = requests + excluded.requests,
				input_tokens = input_tokens + excluded.input_tokens,
				output_tokens = output_tokens + excluded.output_tokens,
				total_tokens = total_tokens + excluded.total_tokens,
				input_tokens_details = excluded.input_tokens_details,
				output_tokens_details = excluded.output_tokens_details,
				updated_at = CURRENT_TIMESTAMP
		`, s.turnUsageTable),
		s.rootSessionID,
		branchID,
		userTurn,
		runUsage.Requests,
		runUsage.InputTokens,
		runUsage.OutputTokens,
		runUsage.TotalTokens,
		string(inputDetailsRaw),
		string(outputDetailsRaw),
	)
	return err
}

func (s *AdvancedSQLiteSession) initTurnUsageTable(ctx context.Context, db *sql.DB) error {
	_, err := db.ExecContext(ctx, fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS "%s" (
			session_id TEXT NOT NULL,
			branch_id TEXT NOT NULL,
			user_turn_number INTEGER NOT NULL,
			requests INTEGER NOT NULL DEFAULT 0,
			input_tokens INTEGER NOT NULL DEFAULT 0,
			output_tokens INTEGER NOT NULL DEFAULT 0,
			total_tokens INTEGER NOT NULL DEFAULT 0,
			input_tokens_details TEXT,
			output_tokens_details TEXT,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (session_id, branch_id, user_turn_number)
		)
	`, s.turnUsageTable))
	if err != nil {
		return fmt.Errorf("failed to initialize turn usage table: %w", err)
	}

	return nil
}

func (s *AdvancedSQLiteSession) newBranchSession(ctx context.Context, branchID string) (*SQLiteSession, error) {
	return NewSQLiteSession(ctx, SQLiteSessionParams{
		SessionID:        branchSessionID(s.rootSessionID, branchID),
		DBDataSourceName: s.dbDSN,
		SessionTable:     s.sessionTable,
		MessagesTable:    s.messagesTable,
	})
}

func branchSessionID(rootSessionID, branchID string) string {
	if branchID == MainBranchID {
		return rootSessionID
	}
	return rootSessionID + "::branch::" + branchID
}

func itemsBeforeTurn(items []TResponseInputItem, turn int) []TResponseInputItem {
	if turn <= 1 {
		return nil
	}

	userTurn := 0
	out := make([]TResponseInputItem, 0, len(items))
	for _, item := range items {
		if isUserMessageItem(item) {
			userTurn++
			if userTurn >= turn {
				break
			}
		}
		out = append(out, item)
	}
	return out
}

func isUserMessageItem(item TResponseInputItem) bool {
	if item.OfMessage != nil {
		return item.OfMessage.Role == responses.EasyInputMessageRoleUser
	}
	if item.OfInputMessage != nil {
		return item.OfInputMessage.Role == "user"
	}
	return false
}

func extractUserMessageContent(item TResponseInputItem) (string, bool) {
	if item.OfMessage != nil && item.OfMessage.Role == responses.EasyInputMessageRoleUser {
		if item.OfMessage.Content.OfString.Valid() {
			return item.OfMessage.Content.OfString.Value, true
		}
		return "", true
	}

	if item.OfInputMessage != nil && item.OfInputMessage.Role == "user" {
		parts := make([]string, 0, len(item.OfInputMessage.Content))
		for _, content := range item.OfInputMessage.Content {
			if text := content.GetText(); text != nil {
				parts = append(parts, *text)
			}
		}
		if len(parts) == 0 {
			return "", true
		}
		return strings.Join(parts, "\n"), true
	}

	return "", false
}

func truncateWithEllipsis(v string, maxLen int) string {
	if maxLen <= 0 || len(v) <= maxLen {
		return v
	}
	return v[:maxLen] + "..."
}

func inputItemType(item TResponseInputItem) string {
	if item.OfMessage != nil {
		return string(item.OfMessage.Role)
	}
	if item.OfInputMessage != nil {
		return item.OfInputMessage.Role
	}
	if item.OfOutputMessage != nil {
		return string(item.OfOutputMessage.Role)
	}
	if itemType := item.GetType(); itemType != nil {
		return *itemType
	}
	return "unknown"
}

func extractToolName(item TResponseInputItem, itemType string) string {
	switch itemType {
	case "function_call":
		if item.OfFunctionCall != nil {
			return item.OfFunctionCall.Name
		}
	case "custom_tool_call":
		if item.OfCustomToolCall != nil {
			return item.OfCustomToolCall.Name
		}
	case "mcp_call":
		if item.OfMcpCall != nil {
			if item.OfMcpCall.ServerLabel != "" && item.OfMcpCall.Name != "" {
				return item.OfMcpCall.ServerLabel + "." + item.OfMcpCall.Name
			}
			if item.OfMcpCall.Name != "" {
				return item.OfMcpCall.Name
			}
		}
	case "mcp_approval_request":
		if item.OfMcpApprovalRequest != nil {
			if item.OfMcpApprovalRequest.ServerLabel != "" && item.OfMcpApprovalRequest.Name != "" {
				return item.OfMcpApprovalRequest.ServerLabel + "." + item.OfMcpApprovalRequest.Name
			}
			if item.OfMcpApprovalRequest.Name != "" {
				return item.OfMcpApprovalRequest.Name
			}
		}
	case "computer_call", "file_search_call", "web_search_call", "code_interpreter_call":
		return itemType
	}
	return ""
}

func isToolCallItemType(itemType string) bool {
	switch itemType {
	case "tool_call",
		"function_call",
		"computer_call",
		"file_search_call",
		"web_search_call",
		"code_interpreter_call",
		"custom_tool_call",
		"mcp_call",
		"mcp_approval_request":
		return true
	default:
		return false
	}
}

func decodeUsageDetails(
	inputDetailsRaw sql.NullString,
	outputDetailsRaw sql.NullString,
) (
	responses.ResponseUsageInputTokensDetails,
	responses.ResponseUsageOutputTokensDetails,
) {
	var inputDetails responses.ResponseUsageInputTokensDetails
	if inputDetailsRaw.Valid && strings.TrimSpace(inputDetailsRaw.String) != "" {
		_ = json.Unmarshal([]byte(inputDetailsRaw.String), &inputDetails)
	}

	var outputDetails responses.ResponseUsageOutputTokensDetails
	if outputDetailsRaw.Valid && strings.TrimSpace(outputDetailsRaw.String) != "" {
		_ = json.Unmarshal([]byte(outputDetailsRaw.String), &outputDetails)
	}

	return inputDetails, outputDetails
}

func maxInt64ToZero(v int64) int64 {
	if v < 0 {
		return 0
	}
	return v
}

func dbTimestampToString(v any) string {
	switch t := v.(type) {
	case time.Time:
		return t.Format(time.RFC3339Nano)
	case string:
		return t
	case []byte:
		return string(t)
	default:
		return fmt.Sprint(t)
	}
}

var _ Session = (*AdvancedSQLiteSession)(nil)
