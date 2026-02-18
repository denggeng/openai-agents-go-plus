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

package agents

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strings"
)

type DocstringStyle string

const (
	DocstringStyleAuto   DocstringStyle = ""
	DocstringStyleGoogle DocstringStyle = "google"
	DocstringStyleNumpy  DocstringStyle = "numpy"
	DocstringStyleSphinx DocstringStyle = "sphinx"
)

type FuncDocumentation struct {
	Name              string
	Description       string
	ParamDescriptions map[string]string
}

func GenerateFuncDocumentation(fn any, style ...DocstringStyle) (FuncDocumentation, error) {
	name, file, err := functionNameAndFile(fn)
	if err != nil {
		return FuncDocumentation{}, err
	}

	doc := extractDocstring(file, name)
	if strings.TrimSpace(doc) == "" {
		return FuncDocumentation{Name: name, Description: "", ParamDescriptions: nil}, nil
	}

	docStyle := DocstringStyleAuto
	if len(style) > 0 {
		docStyle = style[0]
	}
	if docStyle == DocstringStyleAuto {
		docStyle = detectDocstringStyle(doc)
	}

	description, paramDescriptions := parseDocstring(doc, docStyle)
	if len(paramDescriptions) == 0 {
		paramDescriptions = nil
	}

	return FuncDocumentation{
		Name:              name,
		Description:       description,
		ParamDescriptions: paramDescriptions,
	}, nil
}

func functionNameAndFile(fn any) (string, string, error) {
	if fn == nil {
		return "", "", fmt.Errorf("function is nil")
	}
	value := reflect.ValueOf(fn)
	if value.Kind() != reflect.Func {
		return "", "", fmt.Errorf("expected function, got %T", fn)
	}

	pc := value.Pointer()
	runtimeFn := runtime.FuncForPC(pc)
	if runtimeFn == nil {
		return "", "", fmt.Errorf("unable to resolve function")
	}

	fullName := runtimeFn.Name()
	if fullName == "" {
		return "", "", fmt.Errorf("empty function name")
	}

	parts := strings.Split(fullName, ".")
	name := parts[len(parts)-1]
	name = strings.TrimSuffix(name, "-fm")
	if name == "" {
		return "", "", fmt.Errorf("empty function name")
	}

	file, _ := runtimeFn.FileLine(pc)
	if file == "" {
		return "", "", fmt.Errorf("missing source file for %s", fullName)
	}

	return name, file, nil
}

func extractDocstring(filename, funcName string) string {
	if filename == "" || funcName == "" {
		return ""
	}
	if doc := extractDocstringFromFile(filename, funcName); doc != "" {
		return doc
	}
	dir := filepath.Dir(filename)
	if dir == "" {
		return ""
	}
	return extractDocstringFromDir(dir, funcName)
}

func extractDocstringFromFile(filename, funcName string) string {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filename, nil, parser.ParseComments)
	if err != nil || file == nil {
		return ""
	}
	return extractDocstringFromFileAst(file, funcName)
}

func extractDocstringFromDir(dir, funcName string) string {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, dir, func(info os.FileInfo) bool {
		return strings.HasSuffix(info.Name(), ".go")
	}, parser.ParseComments)
	if err != nil {
		return ""
	}
	for _, pkg := range pkgs {
		for _, file := range pkg.Files {
			if doc := extractDocstringFromFileAst(file, funcName); doc != "" {
				return doc
			}
		}
	}
	return ""
}

func extractDocstringFromFileAst(file *ast.File, funcName string) string {
	for _, decl := range file.Decls {
		fnDecl, ok := decl.(*ast.FuncDecl)
		if !ok || fnDecl.Name == nil {
			continue
		}
		if fnDecl.Name.Name != funcName {
			continue
		}
		if fnDecl.Doc == nil {
			return ""
		}
		return strings.TrimSpace(fnDecl.Doc.Text())
	}
	return ""
}

func detectDocstringStyle(doc string) DocstringStyle {
	scores := map[DocstringStyle]int{
		DocstringStyleSphinx: 0,
		DocstringStyleNumpy:  0,
		DocstringStyleGoogle: 0,
	}

	sphinxPatterns := []string{`^:param\s`, `^:type\s`, `^:return:`, `^:rtype:`}
	for _, pattern := range sphinxPatterns {
		if regexp.MustCompile("(?m)"+pattern).FindStringIndex(doc) != nil {
			scores[DocstringStyleSphinx]++
		}
	}

	numpyPatterns := []string{
		`^Parameters\s*\n\s*-{3,}`,
		`^Returns\s*\n\s*-{3,}`,
		`^Yields\s*\n\s*-{3,}`,
	}
	for _, pattern := range numpyPatterns {
		if regexp.MustCompile("(?m)"+pattern).FindStringIndex(doc) != nil {
			scores[DocstringStyleNumpy]++
		}
	}

	googlePatterns := []string{`^(Args|Arguments):`, `^(Returns):`, `^(Raises):`}
	for _, pattern := range googlePatterns {
		if regexp.MustCompile("(?m)"+pattern).FindStringIndex(doc) != nil {
			scores[DocstringStyleGoogle]++
		}
	}

	maxScore := 0
	for _, score := range scores {
		if score > maxScore {
			maxScore = score
		}
	}
	if maxScore == 0 {
		return DocstringStyleGoogle
	}

	for _, style := range []DocstringStyle{DocstringStyleSphinx, DocstringStyleNumpy, DocstringStyleGoogle} {
		if scores[style] == maxScore {
			return style
		}
	}

	return DocstringStyleGoogle
}

func parseDocstring(doc string, style DocstringStyle) (string, map[string]string) {
	lines := strings.Split(strings.ReplaceAll(doc, "\r\n", "\n"), "\n")
	lines = trimEmptyLines(lines)

	description := extractDescription(lines)

	switch style {
	case DocstringStyleSphinx:
		return description, parseSphinxParams(lines)
	case DocstringStyleNumpy:
		return description, parseNumpyParams(lines)
	default:
		return description, parseGoogleParams(lines)
	}
}

func trimEmptyLines(lines []string) []string {
	start := 0
	for start < len(lines) && strings.TrimSpace(lines[start]) == "" {
		start++
	}
	end := len(lines)
	for end > start && strings.TrimSpace(lines[end-1]) == "" {
		end--
	}
	return lines[start:end]
}

func extractDescription(lines []string) string {
	var descriptionLines []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			if len(descriptionLines) > 0 {
				break
			}
			continue
		}
		if isSectionHeader(trimmed) || strings.HasPrefix(trimmed, ":param ") {
			break
		}
		descriptionLines = append(descriptionLines, trimmed)
	}
	return strings.TrimSpace(strings.Join(descriptionLines, "\n"))
}

func isSectionHeader(line string) bool {
	switch line {
	case "Args:", "Arguments:", "Returns:", "Raises:", "Parameters", "Returns", "Yields":
		return true
	default:
		return false
	}
}

func parseSphinxParams(lines []string) map[string]string {
	params := make(map[string]string)
	re := regexp.MustCompile(`^:param\s+(\w+)\s*:\s*(.+)$`)
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		match := re.FindStringSubmatch(trimmed)
		if len(match) == 3 {
			params[match[1]] = strings.TrimSpace(match[2])
		}
	}
	return params
}

func parseGoogleParams(lines []string) map[string]string {
	params := make(map[string]string)
	inArgs := false
	current := ""
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		if isGoogleSectionHeader(trimmed) {
			inArgs = trimmed == "Args:" || trimmed == "Arguments:"
			current = ""
			continue
		}
		if !inArgs {
			continue
		}
		name, desc, ok := parseParamLine(trimmed)
		if ok {
			current = name
			params[current] = desc
			continue
		}
		if current != "" {
			if params[current] != "" {
				params[current] += " "
			}
			params[current] += trimmed
		}
	}
	return params
}

func isGoogleSectionHeader(line string) bool {
	return line == "Args:" || line == "Arguments:" || line == "Returns:" || line == "Raises:"
}

func parseNumpyParams(lines []string) map[string]string {
	params := make(map[string]string)
	inParams := false
	current := ""
	for i := 0; i < len(lines); i++ {
		line := lines[i]
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		if trimmed == "Parameters" && i+1 < len(lines) && isDashLine(lines[i+1]) {
			inParams = true
			current = ""
			i++
			continue
		}
		if inParams && trimmed == "Returns" && i+1 < len(lines) && isDashLine(lines[i+1]) {
			break
		}
		if !inParams {
			continue
		}
		if !strings.HasPrefix(line, " ") && strings.Contains(trimmed, ":") {
			name, _, _ := strings.Cut(trimmed, ":")
			current = strings.TrimSpace(name)
			if current != "" {
				params[current] = ""
			}
			continue
		}
		if current != "" {
			if params[current] != "" {
				params[current] += " "
			}
			params[current] += strings.TrimSpace(trimmed)
		}
	}
	return params
}

func isDashLine(line string) bool {
	return regexp.MustCompile(`^\s*-{3,}\s*$`).FindStringIndex(line) != nil
}

func parseParamLine(line string) (string, string, bool) {
	name, desc, ok := strings.Cut(line, ":")
	if !ok {
		return "", "", false
	}
	name = strings.TrimSpace(name)
	desc = strings.TrimSpace(desc)
	if name == "" || desc == "" {
		return "", "", false
	}
	return name, desc, true
}
