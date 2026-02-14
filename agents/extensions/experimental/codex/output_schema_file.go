// Copyright 2026 The NLP Odyssey Authors
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

package codex

import (
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/nlpodyssey/openai-agents-go/agents"
)

// OutputSchemaFile stores the optional schema file path and cleanup callback.
type OutputSchemaFile struct {
	SchemaPath *string
	cleanup    func()
}

// Cleanup performs best-effort cleanup for generated schema files.
func (o OutputSchemaFile) Cleanup() {
	if o.cleanup != nil {
		o.cleanup()
	}
}

// CreateOutputSchemaFile writes a JSON schema into a temporary file for Codex CLI usage.
func CreateOutputSchemaFile(schema any) (OutputSchemaFile, error) {
	if schema == nil {
		return OutputSchemaFile{
			SchemaPath: nil,
			cleanup:    func() {},
		}, nil
	}

	schemaMap, ok := schema.(map[string]any)
	if !ok {
		return OutputSchemaFile{}, agents.NewUserError("output_schema must be a plain JSON object")
	}

	schemaDir, err := os.MkdirTemp("", "codex-output-schema-")
	if err != nil {
		return OutputSchemaFile{}, err
	}
	schemaPath := filepath.Join(schemaDir, "schema.json")

	cleanup := func() {
		_ = os.RemoveAll(schemaDir)
	}

	file, err := os.Create(schemaPath)
	if err != nil {
		cleanup()
		return OutputSchemaFile{}, err
	}
	encoder := json.NewEncoder(file)
	if err := encoder.Encode(schemaMap); err != nil {
		_ = file.Close()
		cleanup()
		return OutputSchemaFile{}, err
	}
	if err := file.Close(); err != nil {
		cleanup()
		return OutputSchemaFile{}, err
	}

	return OutputSchemaFile{
		SchemaPath: &schemaPath,
		cleanup:    cleanup,
	}, nil
}
