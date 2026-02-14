package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
)

const DefaultCompactionThreshold = 10

// OpenAIResponsesCompactionMode controls how compaction history is provided.
type OpenAIResponsesCompactionMode string

const (
	OpenAIResponsesCompactionModePreviousResponseID OpenAIResponsesCompactionMode = "previous_response_id"
	OpenAIResponsesCompactionModeInput              OpenAIResponsesCompactionMode = "input"
	OpenAIResponsesCompactionModeAuto               OpenAIResponsesCompactionMode = "auto"
)

type resolvedCompactionMode string

const (
	resolvedCompactionModePreviousResponseID resolvedCompactionMode = "previous_response_id"
	resolvedCompactionModeInput              resolvedCompactionMode = "input"
)

// OpenAIResponsesCompactionArgs contains optional run parameters for compaction.
type OpenAIResponsesCompactionArgs struct {
	ResponseID     string
	CompactionMode OpenAIResponsesCompactionMode
	Store          *bool
	Force          bool
}

// OpenAIResponsesCompactionAwareSession marks sessions that support responses compaction.
type OpenAIResponsesCompactionAwareSession interface {
	Session
	RunCompaction(ctx context.Context, args *OpenAIResponsesCompactionArgs) error
}

// IsOpenAIResponsesCompactionAwareSession returns true if session supports compaction.
func IsOpenAIResponsesCompactionAwareSession(session Session) bool {
	if session == nil {
		return false
	}
	_, ok := session.(OpenAIResponsesCompactionAwareSession)
	return ok
}

// CompactionDecisionContext is passed to ShouldTriggerCompaction.
type CompactionDecisionContext struct {
	ResponseID               string
	CompactionMode           string
	CompactionCandidateItems []TResponseInputItem
	SessionItems             []TResponseInputItem
}

type shouldTriggerCompactionFunc func(context CompactionDecisionContext) bool

type responseCompactionService interface {
	Compact(
		ctx context.Context,
		body responses.ResponseCompactParams,
		opts ...option.RequestOption,
	) (*responses.CompactedResponse, error)
}

// OpenAIResponsesCompactionSessionParams configures OpenAIResponsesCompactionSession.
type OpenAIResponsesCompactionSessionParams struct {
	SessionID         string
	UnderlyingSession Session
	Client            *openai.Client
	Model             string
	CompactionMode    OpenAIResponsesCompactionMode

	ShouldTriggerCompaction shouldTriggerCompactionFunc

	// Internal dependency injection hook used in tests.
	CompactionService responseCompactionService
}

// OpenAIResponsesCompactionSession is a Session decorator that compacts long history.
type OpenAIResponsesCompactionSession struct {
	sessionID               string
	underlyingSession       Session
	compactionService       responseCompactionService
	model                   string
	compactionMode          OpenAIResponsesCompactionMode
	shouldTriggerCompaction shouldTriggerCompactionFunc

	compactionCandidateItems []TResponseInputItem
	sessionItems             []TResponseInputItem
	responseID               string
	deferredResponseID       string
	lastUnstoredResponseID   string
	mu                       sync.Mutex
}

// NewOpenAIResponsesCompactionSession creates a compaction-aware session decorator.
func NewOpenAIResponsesCompactionSession(
	params OpenAIResponsesCompactionSessionParams,
) (*OpenAIResponsesCompactionSession, error) {
	if params.UnderlyingSession == nil {
		return nil, errors.New("underlying session is required")
	}
	if _, ok := params.UnderlyingSession.(*OpenAIConversationsSession); ok {
		return nil, errors.New("OpenAIResponsesCompactionSession cannot wrap OpenAIConversationsSession")
	}

	model := strings.TrimSpace(params.Model)
	if model == "" {
		model = "gpt-4.1"
	}
	if !IsOpenAIModelName(model) {
		return nil, fmt.Errorf("unsupported model for OpenAI responses compaction: %s", model)
	}

	mode := params.CompactionMode
	if mode == "" {
		mode = OpenAIResponsesCompactionModeAuto
	}

	shouldTrigger := params.ShouldTriggerCompaction
	if shouldTrigger == nil {
		shouldTrigger = DefaultShouldTriggerCompaction
	}

	service := params.CompactionService
	if service == nil {
		client := params.Client
		if client == nil {
			c := newDefaultOpenAIClient()
			client = &c
		}
		service = &client.Responses
	}

	return &OpenAIResponsesCompactionSession{
		sessionID:               params.SessionID,
		underlyingSession:       params.UnderlyingSession,
		compactionService:       service,
		model:                   model,
		compactionMode:          mode,
		shouldTriggerCompaction: shouldTrigger,
	}, nil
}

func (s *OpenAIResponsesCompactionSession) SessionID(ctx context.Context) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.sessionID != "" {
		return s.sessionID
	}
	return s.underlyingSession.SessionID(ctx)
}

func (s *OpenAIResponsesCompactionSession) GetItems(ctx context.Context, limit int) ([]TResponseInputItem, error) {
	return s.underlyingSession.GetItems(ctx, limit)
}

func (s *OpenAIResponsesCompactionSession) AddItems(ctx context.Context, items []TResponseInputItem) error {
	if err := s.underlyingSession.AddItems(ctx, items); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.compactionCandidateItems != nil {
		newCandidates := SelectCompactionCandidateItems(items)
		if len(newCandidates) > 0 {
			s.compactionCandidateItems = append(s.compactionCandidateItems, newCandidates...)
		}
	}
	if s.sessionItems != nil {
		s.sessionItems = append(s.sessionItems, items...)
	}
	return nil
}

func (s *OpenAIResponsesCompactionSession) PopItem(ctx context.Context) (*TResponseInputItem, error) {
	popped, err := s.underlyingSession.PopItem(ctx)
	if err != nil {
		return nil, err
	}
	if popped != nil {
		s.mu.Lock()
		s.compactionCandidateItems = nil
		s.sessionItems = nil
		s.mu.Unlock()
	}
	return popped, nil
}

func (s *OpenAIResponsesCompactionSession) ClearSession(ctx context.Context) error {
	if err := s.underlyingSession.ClearSession(ctx); err != nil {
		return err
	}

	s.mu.Lock()
	s.compactionCandidateItems = []TResponseInputItem{}
	s.sessionItems = []TResponseInputItem{}
	s.deferredResponseID = ""
	s.mu.Unlock()
	return nil
}

// RunCompaction runs compaction and replaces underlying history with compacted output.
func (s *OpenAIResponsesCompactionSession) RunCompaction(
	ctx context.Context,
	args *OpenAIResponsesCompactionArgs,
) error {
	s.mu.Lock()
	if args != nil && args.ResponseID != "" {
		s.responseID = args.ResponseID
	}
	requestedMode := OpenAIResponsesCompactionMode("")
	var store *bool
	if args != nil {
		requestedMode = args.CompactionMode
		store = args.Store
	}
	if store != nil {
		if !*store && s.responseID != "" {
			s.lastUnstoredResponseID = s.responseID
		} else if *store && s.responseID != "" && s.responseID == s.lastUnstoredResponseID {
			s.lastUnstoredResponseID = ""
		}
	}
	currentResponseID := s.responseID
	resolvedMode := s.resolveCompactionModeForResponse(currentResponseID, store, requestedMode)
	s.mu.Unlock()

	if resolvedMode == resolvedCompactionModePreviousResponseID && currentResponseID == "" {
		return errors.New(
			"OpenAIResponsesCompactionSession.run_compaction requires a response_id " +
				"when using previous_response_id compaction",
		)
	}

	compactionCandidateItems, sessionItems, err := s.ensureCompactionCandidates(ctx)
	if err != nil {
		return err
	}

	force := args != nil && args.Force
	shouldCompact := force || s.shouldTriggerCompaction(CompactionDecisionContext{
		ResponseID:               currentResponseID,
		CompactionMode:           string(resolvedMode),
		CompactionCandidateItems: compactionCandidateItems,
		SessionItems:             sessionItems,
	})
	if !shouldCompact {
		return nil
	}

	params := responses.ResponseCompactParams{
		Model: responses.ResponseCompactParamsModel(s.model),
	}
	if resolvedMode == resolvedCompactionModePreviousResponseID {
		params.PreviousResponseID = param.NewOpt(currentResponseID)
	} else {
		params.Input = responses.ResponseCompactParamsInputUnion{
			OfResponseInputItemArray: sessionItems,
		}
	}

	compacted, err := s.compactionService.Compact(ctx, params)
	if err != nil {
		return err
	}

	if err := s.underlyingSession.ClearSession(ctx); err != nil {
		return err
	}

	outputItems := make([]TResponseInputItem, 0)
	if compacted != nil {
		for _, item := range compacted.Output {
			inputItem, err := responseInputItemFromResponseOutput(item)
			if err != nil {
				return err
			}
			outputItems = append(outputItems, inputItem)
		}
	}

	if len(outputItems) > 0 {
		if err := s.underlyingSession.AddItems(ctx, outputItems); err != nil {
			return err
		}
	}

	s.mu.Lock()
	s.compactionCandidateItems = SelectCompactionCandidateItems(outputItems)
	s.sessionItems = slices.Clone(outputItems)
	s.deferredResponseID = ""
	s.mu.Unlock()
	return nil
}

func (s *OpenAIResponsesCompactionSession) deferCompaction(
	ctx context.Context,
	responseID string,
	store *bool,
) error {
	s.mu.Lock()
	if s.deferredResponseID != "" {
		s.mu.Unlock()
		return nil
	}
	s.mu.Unlock()

	compactionCandidateItems, sessionItems, err := s.ensureCompactionCandidates(ctx)
	if err != nil {
		return err
	}

	resolvedMode := s.resolveCompactionModeForResponse(responseID, store, "")
	shouldCompact := s.shouldTriggerCompaction(CompactionDecisionContext{
		ResponseID:               responseID,
		CompactionMode:           string(resolvedMode),
		CompactionCandidateItems: compactionCandidateItems,
		SessionItems:             sessionItems,
	})
	if shouldCompact {
		s.mu.Lock()
		s.deferredResponseID = responseID
		s.mu.Unlock()
	}
	return nil
}

func (s *OpenAIResponsesCompactionSession) getDeferredCompactionResponseID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.deferredResponseID
}

func (s *OpenAIResponsesCompactionSession) clearDeferredCompaction() {
	s.mu.Lock()
	s.deferredResponseID = ""
	s.mu.Unlock()
}

func (s *OpenAIResponsesCompactionSession) ensureCompactionCandidates(
	ctx context.Context,
) ([]TResponseInputItem, []TResponseInputItem, error) {
	s.mu.Lock()
	if s.compactionCandidateItems != nil && s.sessionItems != nil {
		candidates := slices.Clone(s.compactionCandidateItems)
		sessionItems := slices.Clone(s.sessionItems)
		s.mu.Unlock()
		return candidates, sessionItems, nil
	}
	s.mu.Unlock()

	history, err := s.underlyingSession.GetItems(ctx, 0)
	if err != nil {
		return nil, nil, err
	}
	candidates := SelectCompactionCandidateItems(history)

	s.mu.Lock()
	s.compactionCandidateItems = slices.Clone(candidates)
	s.sessionItems = slices.Clone(history)
	s.mu.Unlock()
	return candidates, history, nil
}

func (s *OpenAIResponsesCompactionSession) resolveCompactionModeForResponse(
	responseID string,
	store *bool,
	requestedMode OpenAIResponsesCompactionMode,
) resolvedCompactionMode {
	mode := requestedMode
	if mode == "" {
		mode = s.compactionMode
	}

	s.mu.Lock()
	lastUnstored := s.lastUnstoredResponseID
	s.mu.Unlock()

	if mode == OpenAIResponsesCompactionModeAuto &&
		store == nil &&
		responseID != "" &&
		responseID == lastUnstored {
		return resolvedCompactionModeInput
	}
	return resolveCompactionMode(mode, responseID, store)
}

func resolveCompactionMode(
	requestedMode OpenAIResponsesCompactionMode,
	responseID string,
	store *bool,
) resolvedCompactionMode {
	if requestedMode == OpenAIResponsesCompactionModeInput {
		return resolvedCompactionModeInput
	}
	if requestedMode == OpenAIResponsesCompactionModePreviousResponseID {
		return resolvedCompactionModePreviousResponseID
	}
	if store != nil && !*store {
		return resolvedCompactionModeInput
	}
	if responseID == "" {
		return resolvedCompactionModeInput
	}
	return resolvedCompactionModePreviousResponseID
}

// SelectCompactionCandidateItems excludes user messages and prior compaction items.
func SelectCompactionCandidateItems(items []TResponseInputItem) []TResponseInputItem {
	out := make([]TResponseInputItem, 0, len(items))
	for _, item := range items {
		itemType := item.GetType()
		if itemType != nil && *itemType == "compaction" {
			continue
		}
		role := item.GetRole()
		if role != nil && *role == "user" {
			continue
		}
		out = append(out, item)
	}
	return out
}

// DefaultShouldTriggerCompaction compacts when candidate count reaches threshold.
func DefaultShouldTriggerCompaction(context CompactionDecisionContext) bool {
	return len(context.CompactionCandidateItems) >= DefaultCompactionThreshold
}

// IsOpenAIModelName validates model names accepted for compaction.
func IsOpenAIModelName(model string) bool {
	trimmed := strings.TrimSpace(model)
	if trimmed == "" {
		return false
	}

	withoutFTPrefix := trimmed
	if strings.HasPrefix(withoutFTPrefix, "ft:") {
		withoutFTPrefix = withoutFTPrefix[3:]
	}
	root := withoutFTPrefix
	if i := strings.Index(root, ":"); i >= 0 {
		root = root[:i]
	}

	if strings.HasPrefix(root, "gpt-") {
		return true
	}
	if len(root) >= 2 && root[0] == 'o' && root[1] >= '0' && root[1] <= '9' {
		return true
	}
	return false
}

func responseInputItemFromResponseOutput(output responses.ResponseOutputItemUnion) (TResponseInputItem, error) {
	raw := output.RawJSON()
	if raw == "" {
		jsonBytes, err := json.Marshal(output)
		if err != nil {
			return TResponseInputItem{}, fmt.Errorf("failed to marshal compacted output item: %w", err)
		}
		raw = string(jsonBytes)
	}

	var item TResponseInputItem
	if err := json.Unmarshal([]byte(raw), &item); err != nil {
		return TResponseInputItem{}, fmt.Errorf("failed to convert compacted output item: %w", err)
	}
	return item, nil
}
