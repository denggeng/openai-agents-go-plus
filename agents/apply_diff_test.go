package agents

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestApplyDiffWithFloatingHunkAddsLines(t *testing.T) {
	diff := "@@\n+hello\n+world" // no trailing newline
	got, err := ApplyDiff("", diff, ApplyDiffModeDefault)
	require.NoError(t, err)
	assert.Equal(t, "hello\nworld\n", got)
}

func TestApplyDiffWithEmptyInputAndCRLFDiffPreservesCRLF(t *testing.T) {
	diff := "@@\r\n+hello\r\n+world"
	got, err := ApplyDiff("", diff, ApplyDiffModeDefault)
	require.NoError(t, err)
	assert.Equal(t, "hello\r\nworld\r\n", got)
}

func TestApplyDiffCreateModeRequiresPlusPrefix(t *testing.T) {
	_, err := ApplyDiff("", "plain line", ApplyDiffModeCreate)
	require.Error(t, err)
}

func TestApplyDiffCreateModePerservesTrailingNewline(t *testing.T) {
	diff := "+hello\n+world\n+"
	got, err := ApplyDiff("", diff, ApplyDiffModeCreate)
	require.NoError(t, err)
	assert.Equal(t, "hello\nworld\n", got)
}

func TestApplyDiffAppliesContextualReplacement(t *testing.T) {
	inputText := "line1\nline2\nline3\n"
	diff := "@@ line1\n-line2\n+updated\n line3"
	got, err := ApplyDiff(inputText, diff, ApplyDiffModeDefault)
	require.NoError(t, err)
	assert.Equal(t, "line1\nupdated\nline3\n", got)
}

func TestApplyDiffRaisesOnContextMismatch(t *testing.T) {
	inputText := "one\ntwo\n"
	diff := "@@ -1,2 +1,2 @@\n x\n-two\n+2"
	_, err := ApplyDiff(inputText, diff, ApplyDiffModeDefault)
	require.Error(t, err)
}

func TestApplyDiffWithCRLFInputAndLFDiffPreservesCRLF(t *testing.T) {
	inputText := "line1\r\nline2\r\nline3\r\n"
	diff := "@@ line1\n-line2\n+updated\n line3"
	got, err := ApplyDiff(inputText, diff, ApplyDiffModeDefault)
	require.NoError(t, err)
	assert.Equal(t, "line1\r\nupdated\r\nline3\r\n", got)
}

func TestApplyDiffWithLFInputAndCRLFDiffPreservesLF(t *testing.T) {
	inputText := "line1\nline2\nline3\n"
	diff := "@@ line1\r\n-line2\r\n+updated\r\n line3"
	got, err := ApplyDiff(inputText, diff, ApplyDiffModeDefault)
	require.NoError(t, err)
	assert.Equal(t, "line1\nupdated\nline3\n", got)
}

func TestApplyDiffWithCRLFInputAndCRLFDiffPreservesCRLF(t *testing.T) {
	inputText := "line1\r\nline2\r\nline3\r\n"
	diff := "@@ line1\r\n-line2\r\n+updated\r\n line3"
	got, err := ApplyDiff(inputText, diff, ApplyDiffModeDefault)
	require.NoError(t, err)
	assert.Equal(t, "line1\r\nupdated\r\nline3\r\n", got)
}

func TestApplyDiffCreateModePreservesCRLFNewlines(t *testing.T) {
	diff := "+hello\r\n+world\r\n+"
	got, err := ApplyDiff("", diff, ApplyDiffModeCreate)
	require.NoError(t, err)
	assert.Equal(t, "hello\r\nworld\r\n", got)
}

func TestNormalizeDiffLinesDropsTrailingBlank(t *testing.T) {
	assert.Equal(t, []string{"a", "b"}, normalizeDiffLines("a\nb\n"))
}

func TestIsDoneTrueWhenIndexOutOfRange(t *testing.T) {
	state := parserState{lines: []string{"line"}, index: 1}
	assert.True(t, isDone(&state, nil))
}

func TestReadStrReturnsEmptyWhenMissingPrefix(t *testing.T) {
	state := parserState{lines: []string{"value"}, index: 0}
	assert.Equal(t, "", readStr(&state, "nomatch"))
	assert.Equal(t, 0, state.index)
}

func TestReadSectionReturnsEOFFlag(t *testing.T) {
	result, err := readSection([]string{endOfFileMarker}, 0)
	require.NoError(t, err)
	assert.True(t, result.eof)
}

func TestReadSectionRaisesOnInvalidMarker(t *testing.T) {
	_, err := readSection([]string{"*** Bad Marker"}, 0)
	require.Error(t, err)
}

func TestReadSectionRaisesWhenEmptySegment(t *testing.T) {
	_, err := readSection([]string{}, 0)
	require.Error(t, err)
}

func TestFindContextEOFFallbacks(t *testing.T) {
	match := findContext([]string{"one"}, []string{"missing"}, 0, true)
	assert.Equal(t, -1, match.newIndex)
	assert.GreaterOrEqual(t, match.fuzz, 10000)
}

func TestFindContextCoreStrippedMatches(t *testing.T) {
	match := findContextCore([]string{" line "}, []string{"line"}, 0)
	assert.Equal(t, 0, match.newIndex)
	assert.Equal(t, 100, match.fuzz)
}

func TestApplyChunksRejectsBadChunks(t *testing.T) {
	_, err := applyChunks("abc", []chunk{{origIndex: 10}}, "\n")
	require.Error(t, err)

	_, err = applyChunks("abc", []chunk{
		{origIndex: 0, delLines: []string{"a"}},
		{origIndex: 0, delLines: []string{"b"}},
	}, "\n")
	require.Error(t, err)
}
