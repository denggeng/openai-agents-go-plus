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

package agents_test

import (
	"testing"

	agents "github.com/denggeng/openai-agents-go-plus/agents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// This is func_foo.
//
// Args:
//
//	a: The first argument.
//	b: The second argument.
//
// Returns:
//
//	A result
func func_foo_google(a int, b float64) string {
	return "ok"
}

// This is func_foo.
//
// Parameters
// ----------
// a: int
//
//	The first argument.
//
// b: float
//
//	The second argument.
//
// Returns
// -------
// str
//
//	A result
func func_foo_numpy(a int, b float64) string {
	return "ok"
}

// This is func_foo.
//
// :param a: The first argument.
// :param b: The second argument.
// :return: A result
func func_foo_sphinx(a int, b float64) string {
	return "ok"
}

type Bar struct{}

// This is func_bar.
//
// Args:
//
//	a: The first argument.
//	b: The second argument.
//
// Returns:
//
//	A result
func (Bar) func_bar(a int, b float64) string {
	return "ok"
}

// This is func_baz.
//
// Args:
//
//	a: The first argument.
//	b: The second argument.
//
// Returns:
//
//	A result
func (Bar) func_baz(a int, b float64) string {
	return "ok"
}

func TestFunctionsAreOK(t *testing.T) {
	func_foo_google(1, 2.0)
	func_foo_numpy(1, 2.0)
	func_foo_sphinx(1, 2.0)
	Bar{}.func_bar(1, 2.0)
	Bar{}.func_baz(1, 2.0)
}

func TestAutoDetection(t *testing.T) {
	doc, err := agents.GenerateFuncDocumentation(func_foo_google)
	require.NoError(t, err)
	assert.Equal(t, "func_foo_google", doc.Name)
	assert.Equal(t, "This is func_foo.", doc.Description)
	assert.Equal(t, map[string]string{
		"a": "The first argument.",
		"b": "The second argument.",
	}, doc.ParamDescriptions)

	doc, err = agents.GenerateFuncDocumentation(func_foo_numpy)
	require.NoError(t, err)
	assert.Equal(t, "func_foo_numpy", doc.Name)
	assert.Equal(t, "This is func_foo.", doc.Description)
	assert.Equal(t, map[string]string{
		"a": "The first argument.",
		"b": "The second argument.",
	}, doc.ParamDescriptions)

	doc, err = agents.GenerateFuncDocumentation(func_foo_sphinx)
	require.NoError(t, err)
	assert.Equal(t, "func_foo_sphinx", doc.Name)
	assert.Equal(t, "This is func_foo.", doc.Description)
	assert.Equal(t, map[string]string{
		"a": "The first argument.",
		"b": "The second argument.",
	}, doc.ParamDescriptions)
}

func TestInstanceMethod(t *testing.T) {
	doc, err := agents.GenerateFuncDocumentation(Bar{}.func_bar)
	require.NoError(t, err)
	assert.Equal(t, "func_bar", doc.Name)
	assert.Equal(t, "This is func_bar.", doc.Description)
	assert.Equal(t, map[string]string{
		"a": "The first argument.",
		"b": "The second argument.",
	}, doc.ParamDescriptions)
}

func TestClassmethod(t *testing.T) {
	doc, err := agents.GenerateFuncDocumentation(Bar{}.func_baz)
	require.NoError(t, err)
	assert.Equal(t, "func_baz", doc.Name)
	assert.Equal(t, "This is func_baz.", doc.Description)
	assert.Equal(t, map[string]string{
		"a": "The first argument.",
		"b": "The second argument.",
	}, doc.ParamDescriptions)
}
