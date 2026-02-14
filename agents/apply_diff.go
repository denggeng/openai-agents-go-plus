package agents

import (
	"fmt"
	"strings"
	"unicode"
)

// ApplyDiffMode defines the parser mode used by ApplyDiff.
type ApplyDiffMode string

const (
	ApplyDiffModeDefault ApplyDiffMode = "default"
	ApplyDiffModeCreate  ApplyDiffMode = "create"
)

const (
	endPatchMarker  = "*** End Patch"
	endOfFileMarker = "*** End of File"
)

var (
	sectionTerminators = []string{
		endPatchMarker,
		"*** Update File:",
		"*** Delete File:",
		"*** Add File:",
	}
	endSectionMarkers = []string{
		endPatchMarker,
		"*** Update File:",
		"*** Delete File:",
		"*** Add File:",
		endOfFileMarker,
	}
)

type chunk struct {
	origIndex int
	delLines  []string
	insLines  []string
}

type parserState struct {
	lines []string
	index int
	fuzz  int
}

type parsedUpdateDiff struct {
	chunks []chunk
	fuzz   int
}

type readSectionResult struct {
	nextContext   []string
	sectionChunks []chunk
	endIndex      int
	eof           bool
}

type contextMatch struct {
	newIndex int
	fuzz     int
}

// ApplyDiff applies a V4A diff to the provided input text.
func ApplyDiff(input string, diff string, mode ApplyDiffMode) (string, error) {
	if mode == "" {
		mode = ApplyDiffModeDefault
	}

	newline := detectNewline(input, diff, mode)
	diffLines := normalizeDiffLines(diff)
	if mode == ApplyDiffModeCreate {
		return parseCreateDiff(diffLines, newline)
	}

	normalizedInput := normalizeTextNewlines(input)
	parsed, err := parseUpdateDiff(diffLines, normalizedInput)
	if err != nil {
		return "", err
	}
	return applyChunks(normalizedInput, parsed.chunks, newline)
}

func normalizeDiffLines(diff string) []string {
	lines := strings.Split(diff, "\n")
	for i := range lines {
		lines[i] = strings.TrimSuffix(lines[i], "\r")
	}
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

func detectNewlineFromText(text string) string {
	if strings.Contains(text, "\r\n") {
		return "\r\n"
	}
	return "\n"
}

func detectNewline(input string, diff string, mode ApplyDiffMode) string {
	if mode != ApplyDiffModeCreate && strings.Contains(input, "\n") {
		return detectNewlineFromText(input)
	}
	return detectNewlineFromText(diff)
}

func normalizeTextNewlines(text string) string {
	return strings.ReplaceAll(text, "\r\n", "\n")
}

func isDone(state *parserState, prefixes []string) bool {
	if state.index >= len(state.lines) {
		return true
	}
	current := state.lines[state.index]
	for _, prefix := range prefixes {
		if strings.HasPrefix(current, prefix) {
			return true
		}
	}
	return false
}

func readStr(state *parserState, prefix string) string {
	if state.index >= len(state.lines) {
		return ""
	}
	current := state.lines[state.index]
	if strings.HasPrefix(current, prefix) {
		state.index++
		return current[len(prefix):]
	}
	return ""
}

func parseCreateDiff(lines []string, newline string) (string, error) {
	parser := parserState{
		lines: append(append([]string(nil), lines...), endPatchMarker),
	}
	output := make([]string, 0, len(lines))

	for !isDone(&parser, sectionTerminators) {
		if parser.index >= len(parser.lines) {
			break
		}
		line := parser.lines[parser.index]
		parser.index++
		if !strings.HasPrefix(line, "+") {
			return "", fmt.Errorf("invalid Add File Line: %s", line)
		}
		output = append(output, line[1:])
	}

	return strings.Join(output, newline), nil
}

func parseUpdateDiff(lines []string, input string) (parsedUpdateDiff, error) {
	parser := parserState{
		lines: append(append([]string(nil), lines...), endPatchMarker),
	}
	inputLines := strings.Split(input, "\n")
	chunks := make([]chunk, 0)
	cursor := 0

	for !isDone(&parser, endSectionMarkers) {
		anchor := readStr(&parser, "@@ ")
		hasBareAnchor := anchor == "" && parser.index < len(parser.lines) && parser.lines[parser.index] == "@@"
		if hasBareAnchor {
			parser.index++
		}

		hasAnchor := strings.TrimSpace(anchor) != ""
		if !(hasAnchor || hasBareAnchor || cursor == 0) {
			currentLine := ""
			if parser.index < len(parser.lines) {
				currentLine = parser.lines[parser.index]
			}
			return parsedUpdateDiff{}, fmt.Errorf("invalid Line:\n%s", currentLine)
		}

		if hasAnchor {
			cursor = advanceCursorToAnchor(anchor, inputLines, cursor, &parser)
		}

		section, err := readSection(parser.lines, parser.index)
		if err != nil {
			return parsedUpdateDiff{}, err
		}

		findResult := findContext(inputLines, section.nextContext, cursor, section.eof)
		if findResult.newIndex == -1 {
			ctxText := strings.Join(section.nextContext, "\n")
			if section.eof {
				return parsedUpdateDiff{}, fmt.Errorf("invalid EOF Context %d:\n%s", cursor, ctxText)
			}
			return parsedUpdateDiff{}, fmt.Errorf("invalid Context %d:\n%s", cursor, ctxText)
		}

		cursor = findResult.newIndex + len(section.nextContext)
		parser.fuzz += findResult.fuzz
		parser.index = section.endIndex

		for _, ch := range section.sectionChunks {
			chunks = append(chunks, chunk{
				origIndex: ch.origIndex + findResult.newIndex,
				delLines:  append([]string(nil), ch.delLines...),
				insLines:  append([]string(nil), ch.insLines...),
			})
		}
	}

	return parsedUpdateDiff{chunks: chunks, fuzz: parser.fuzz}, nil
}

func advanceCursorToAnchor(anchor string, inputLines []string, cursor int, parser *parserState) int {
	found := false

	if !anyStringMatches(inputLines[:min(cursor, len(inputLines))], func(line string) bool { return line == anchor }) {
		for i := cursor; i < len(inputLines); i++ {
			if inputLines[i] == anchor {
				cursor = i + 1
				found = true
				break
			}
		}
	}

	if !found && !anyStringMatches(inputLines[:min(cursor, len(inputLines))], func(line string) bool {
		return strings.TrimSpace(line) == strings.TrimSpace(anchor)
	}) {
		for i := cursor; i < len(inputLines); i++ {
			if strings.TrimSpace(inputLines[i]) == strings.TrimSpace(anchor) {
				cursor = i + 1
				parser.fuzz++
				break
			}
		}
	}

	return cursor
}

func anyStringMatches(lines []string, predicate func(string) bool) bool {
	for _, line := range lines {
		if predicate(line) {
			return true
		}
	}
	return false
}

func readSection(lines []string, startIndex int) (readSectionResult, error) {
	contextLines := make([]string, 0)
	delLines := make([]string, 0)
	insLines := make([]string, 0)
	sectionChunks := make([]chunk, 0)
	mode := "keep"
	index := startIndex
	origIndex := index

	for index < len(lines) {
		raw := lines[index]
		if strings.HasPrefix(raw, "@@") ||
			strings.HasPrefix(raw, endPatchMarker) ||
			strings.HasPrefix(raw, "*** Update File:") ||
			strings.HasPrefix(raw, "*** Delete File:") ||
			strings.HasPrefix(raw, "*** Add File:") ||
			strings.HasPrefix(raw, endOfFileMarker) {
			break
		}
		if raw == "***" {
			break
		}
		if strings.HasPrefix(raw, "***") {
			return readSectionResult{}, fmt.Errorf("invalid Line: %s", raw)
		}

		index++
		lastMode := mode
		line := raw
		if line == "" {
			line = " "
		}

		prefix := line[0]
		switch prefix {
		case '+':
			mode = "add"
		case '-':
			mode = "delete"
		case ' ':
			mode = "keep"
		default:
			return readSectionResult{}, fmt.Errorf("invalid Line: %s", line)
		}

		lineContent := line[1:]
		switchingToContext := mode == "keep" && lastMode != mode
		if switchingToContext && (len(delLines) > 0 || len(insLines) > 0) {
			sectionChunks = append(sectionChunks, chunk{
				origIndex: len(contextLines) - len(delLines),
				delLines:  append([]string(nil), delLines...),
				insLines:  append([]string(nil), insLines...),
			})
			delLines = delLines[:0]
			insLines = insLines[:0]
		}

		if mode == "delete" {
			delLines = append(delLines, lineContent)
			contextLines = append(contextLines, lineContent)
		} else if mode == "add" {
			insLines = append(insLines, lineContent)
		} else {
			contextLines = append(contextLines, lineContent)
		}
	}

	if len(delLines) > 0 || len(insLines) > 0 {
		sectionChunks = append(sectionChunks, chunk{
			origIndex: len(contextLines) - len(delLines),
			delLines:  append([]string(nil), delLines...),
			insLines:  append([]string(nil), insLines...),
		})
	}

	if index < len(lines) && lines[index] == endOfFileMarker {
		return readSectionResult{
			nextContext:   contextLines,
			sectionChunks: sectionChunks,
			endIndex:      index + 1,
			eof:           true,
		}, nil
	}

	if index == origIndex {
		nextLine := ""
		if index < len(lines) {
			nextLine = lines[index]
		}
		return readSectionResult{}, fmt.Errorf("Nothing in this section - index=%d %s", index, nextLine)
	}

	return readSectionResult{
		nextContext:   contextLines,
		sectionChunks: sectionChunks,
		endIndex:      index,
		eof:           false,
	}, nil
}

func findContext(lines []string, context []string, start int, eof bool) contextMatch {
	if eof {
		endStart := max(0, len(lines)-len(context))
		endMatch := findContextCore(lines, context, endStart)
		if endMatch.newIndex != -1 {
			return endMatch
		}
		fallback := findContextCore(lines, context, start)
		return contextMatch{
			newIndex: fallback.newIndex,
			fuzz:     fallback.fuzz + 10000,
		}
	}
	return findContextCore(lines, context, start)
}

func findContextCore(lines []string, context []string, start int) contextMatch {
	if len(context) == 0 {
		return contextMatch{
			newIndex: start,
			fuzz:     0,
		}
	}

	for i := start; i < len(lines); i++ {
		if equalsSlice(lines, context, i, func(value string) string { return value }) {
			return contextMatch{newIndex: i, fuzz: 0}
		}
	}
	for i := start; i < len(lines); i++ {
		if equalsSlice(lines, context, i, func(value string) string {
			return strings.TrimRightFunc(value, unicode.IsSpace)
		}) {
			return contextMatch{newIndex: i, fuzz: 1}
		}
	}
	for i := start; i < len(lines); i++ {
		if equalsSlice(lines, context, i, strings.TrimSpace) {
			return contextMatch{newIndex: i, fuzz: 100}
		}
	}

	return contextMatch{newIndex: -1, fuzz: 0}
}

func equalsSlice(source []string, target []string, start int, mapFn func(string) string) bool {
	if start+len(target) > len(source) {
		return false
	}
	for offset, targetValue := range target {
		if mapFn(source[start+offset]) != mapFn(targetValue) {
			return false
		}
	}
	return true
}

func applyChunks(input string, chunks []chunk, newline string) (string, error) {
	origLines := strings.Split(input, "\n")
	destLines := make([]string, 0, len(origLines))
	cursor := 0

	for _, chunk := range chunks {
		if chunk.origIndex > len(origLines) {
			return "", fmt.Errorf(
				"applyDiff: chunk.origIndex %d > input length %d",
				chunk.origIndex, len(origLines),
			)
		}
		if cursor > chunk.origIndex {
			return "", fmt.Errorf(
				"applyDiff: overlapping chunk at %d (cursor %d)",
				chunk.origIndex, cursor,
			)
		}

		destLines = append(destLines, origLines[cursor:chunk.origIndex]...)
		cursor = chunk.origIndex
		if len(chunk.insLines) > 0 {
			destLines = append(destLines, chunk.insLines...)
		}
		cursor += len(chunk.delLines)
	}

	destLines = append(destLines, origLines[cursor:]...)
	return strings.Join(destLines, newline), nil
}
