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

package agents

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEnsureStrictJSONSchemaOneOfConvertedToAnyOf(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"value": map[string]any{
				"oneOf": []any{
					map[string]any{"type": "string"},
					map[string]any{"type": "integer"},
				},
			},
		},
	}

	result, err := EnsureStrictJSONSchema(schema)
	require.NoError(t, err)

	expected := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"value": map[string]any{
				"anyOf": []any{
					map[string]any{"type": "string"},
					map[string]any{"type": "integer"},
				},
			},
		},
		"additionalProperties": false,
		"required":             []any{"value"},
	}
	assert.Equal(t, expected, result)
}

func TestEnsureStrictJSONSchemaNestedOneOfInArrayItems(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"steps": map[string]any{
				"type": "array",
				"items": map[string]any{
					"oneOf": []any{
						map[string]any{
							"type": "object",
							"properties": map[string]any{
								"action": map[string]any{"type": "string", "const": "buy_fruit"},
								"color":  map[string]any{"type": "string"},
							},
							"required": []any{"action", "color"},
						},
						map[string]any{
							"type": "object",
							"properties": map[string]any{
								"action": map[string]any{"type": "string", "const": "buy_food"},
								"price":  map[string]any{"type": "integer"},
							},
							"required": []any{"action", "price"},
						},
					},
					"discriminator": map[string]any{
						"propertyName": "action",
						"mapping": map[string]any{
							"buy_fruit": "#/components/schemas/BuyFruitStep",
							"buy_food":  "#/components/schemas/BuyFoodStep",
						},
					},
				},
			},
		},
	}

	result, err := EnsureStrictJSONSchema(schema)
	require.NoError(t, err)

	expected := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"steps": map[string]any{
				"type": "array",
				"items": map[string]any{
					"anyOf": []any{
						map[string]any{
							"type": "object",
							"properties": map[string]any{
								"action": map[string]any{"type": "string", "const": "buy_fruit"},
								"color":  map[string]any{"type": "string"},
							},
							"required":             []any{"action", "color"},
							"additionalProperties": false,
						},
						map[string]any{
							"type": "object",
							"properties": map[string]any{
								"action": map[string]any{"type": "string", "const": "buy_food"},
								"price":  map[string]any{"type": "integer"},
							},
							"required":             []any{"action", "price"},
							"additionalProperties": false,
						},
					},
					"discriminator": map[string]any{
						"propertyName": "action",
						"mapping": map[string]any{
							"buy_fruit": "#/components/schemas/BuyFruitStep",
							"buy_food":  "#/components/schemas/BuyFoodStep",
						},
					},
				},
			},
		},
		"additionalProperties": false,
		"required":             []any{"steps"},
	}
	assert.Equal(t, expected, result)
}

func TestEnsureStrictJSONSchemaOneOfMergedWithExistingAnyOf(t *testing.T) {
	schema := map[string]any{
		"type":  "object",
		"anyOf": []any{map[string]any{"type": "string"}},
		"oneOf": []any{
			map[string]any{"type": "integer"},
			map[string]any{"type": "boolean"},
		},
	}

	result, err := EnsureStrictJSONSchema(schema)
	require.NoError(t, err)

	expected := map[string]any{
		"type": "object",
		"anyOf": []any{
			map[string]any{"type": "string"},
			map[string]any{"type": "integer"},
			map[string]any{"type": "boolean"},
		},
		"additionalProperties": false,
	}
	assert.Equal(t, expected, result)
}

func TestEnsureStrictJSONSchemaDiscriminatorPreserved(t *testing.T) {
	schema := map[string]any{
		"oneOf": []any{
			map[string]any{"$ref": "#/$defs/TypeA"},
			map[string]any{"$ref": "#/$defs/TypeB"},
		},
		"discriminator": map[string]any{
			"propertyName": "type",
			"mapping": map[string]any{
				"a": "#/$defs/TypeA",
				"b": "#/$defs/TypeB",
			},
		},
		"$defs": map[string]any{
			"TypeA": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"type":    map[string]any{"const": "a"},
					"value_a": map[string]any{"type": "string"},
				},
			},
			"TypeB": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"type":    map[string]any{"const": "b"},
					"value_b": map[string]any{"type": "integer"},
				},
			},
		},
	}

	result, err := EnsureStrictJSONSchema(schema)
	require.NoError(t, err)

	expected := map[string]any{
		"anyOf": []any{
			map[string]any{"$ref": "#/$defs/TypeA"},
			map[string]any{"$ref": "#/$defs/TypeB"},
		},
		"discriminator": map[string]any{
			"propertyName": "type",
			"mapping": map[string]any{
				"a": "#/$defs/TypeA",
				"b": "#/$defs/TypeB",
			},
		},
		"$defs": map[string]any{
			"TypeA": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"type":    map[string]any{"const": "a"},
					"value_a": map[string]any{"type": "string"},
				},
				"additionalProperties": false,
				"required":             []any{"type", "value_a"},
			},
			"TypeB": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"type":    map[string]any{"const": "b"},
					"value_b": map[string]any{"type": "integer"},
				},
				"additionalProperties": false,
				"required":             []any{"type", "value_b"},
			},
		},
	}
	assert.Equal(t, expected, result)
}

func TestEnsureStrictJSONSchemaDeeplyNestedOneOf(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"level1": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"level2": map[string]any{
						"type": "array",
						"items": map[string]any{
							"oneOf": []any{
								map[string]any{"type": "string"},
								map[string]any{"type": "number"},
							},
						},
					},
				},
			},
		},
	}

	result, err := EnsureStrictJSONSchema(schema)
	require.NoError(t, err)

	expected := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"level1": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"level2": map[string]any{
						"type": "array",
						"items": map[string]any{
							"anyOf": []any{
								map[string]any{"type": "string"},
								map[string]any{"type": "number"},
							},
						},
					},
				},
				"additionalProperties": false,
				"required":             []any{"level2"},
			},
		},
		"additionalProperties": false,
		"required":             []any{"level1"},
	}
	assert.Equal(t, expected, result)
}

func TestEnsureStrictJSONSchemaOneOfWithRefs(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"value": map[string]any{
				"oneOf": []any{
					map[string]any{"$ref": "#/$defs/StringType"},
					map[string]any{"$ref": "#/$defs/IntType"},
				},
			},
		},
		"$defs": map[string]any{
			"StringType": map[string]any{"type": "string"},
			"IntType":    map[string]any{"type": "integer"},
		},
	}

	result, err := EnsureStrictJSONSchema(schema)
	require.NoError(t, err)

	expected := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"value": map[string]any{
				"anyOf": []any{
					map[string]any{"$ref": "#/$defs/StringType"},
					map[string]any{"$ref": "#/$defs/IntType"},
				},
			},
		},
		"$defs": map[string]any{
			"StringType": map[string]any{"type": "string"},
			"IntType":    map[string]any{"type": "integer"},
		},
		"additionalProperties": false,
		"required":             []any{"value"},
	}
	assert.Equal(t, expected, result)
}
