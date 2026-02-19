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

import "reflect"

func materializeJSONMap(value map[string]any) map[string]any {
	if value == nil {
		return nil
	}
	out := make(map[string]any, len(value))
	for key, item := range value {
		out[key] = materializeJSONValue(item)
	}
	return out
}

func materializeJSONValue(value any) any {
	if value == nil {
		return nil
	}
	if seq, ok := materializeSequence(value); ok {
		return seq
	}

	switch typed := value.(type) {
	case map[string]any:
		return materializeJSONMap(typed)
	case []any:
		out := make([]any, len(typed))
		for i, item := range typed {
			out[i] = materializeJSONValue(item)
		}
		return out
	default:
		return materializeJSONValueReflect(typed)
	}
}

func materializeJSONValueReflect(value any) any {
	rv := reflect.ValueOf(value)
	if !rv.IsValid() {
		return nil
	}
	if rv.Kind() == reflect.Pointer {
		if rv.IsNil() {
			return nil
		}
		return materializeJSONValue(rv.Elem().Interface())
	}
	switch rv.Kind() {
	case reflect.Slice, reflect.Array:
		out := reflect.MakeSlice(rv.Type(), rv.Len(), rv.Len())
		elemType := rv.Type().Elem()
		for i := 0; i < rv.Len(); i++ {
			item := materializeJSONValue(rv.Index(i).Interface())
			iv := reflect.ValueOf(item)
			if !iv.IsValid() {
				iv = reflect.Zero(elemType)
			} else if !iv.Type().AssignableTo(elemType) {
				if iv.Type().ConvertibleTo(elemType) {
					iv = iv.Convert(elemType)
				} else {
					iv = rv.Index(i)
				}
			}
			out.Index(i).Set(iv)
		}
		return out.Interface()
	case reflect.Map:
		keyType := rv.Type().Key()
		valType := rv.Type().Elem()
		out := reflect.MakeMapWithSize(rv.Type(), rv.Len())
		iter := rv.MapRange()
		for iter.Next() {
			key := iter.Key()
			val := materializeJSONValue(iter.Value().Interface())
			vv := reflect.ValueOf(val)
			if !vv.IsValid() {
				vv = reflect.Zero(valType)
			} else if !vv.Type().AssignableTo(valType) {
				if vv.Type().ConvertibleTo(valType) {
					vv = vv.Convert(valType)
				} else {
					vv = iter.Value()
				}
			}
			if key.Type() != keyType {
				if key.Type().ConvertibleTo(keyType) {
					key = key.Convert(keyType)
				} else {
					continue
				}
			}
			out.SetMapIndex(key, vv)
		}
		return out.Interface()
	default:
		return value
	}
}

func materializeSequence(value any) (any, bool) {
	rv := reflect.ValueOf(value)
	if !rv.IsValid() || rv.Kind() != reflect.Func {
		return nil, false
	}
	rt := rv.Type()
	if rt.NumIn() != 1 || rt.NumOut() != 0 {
		return nil, false
	}
	yieldType := rt.In(0)
	if yieldType.Kind() != reflect.Func || yieldType.NumOut() != 1 || yieldType.Out(0).Kind() != reflect.Bool {
		return nil, false
	}
	if yieldType.NumIn() < 1 || yieldType.NumIn() > 2 {
		return nil, false
	}

	yieldInputs := yieldType.NumIn()
	if yieldInputs == 1 {
		elemType := yieldType.In(0)
		items := reflect.MakeSlice(reflect.SliceOf(elemType), 0, 0)
		yieldFunc := reflect.MakeFunc(yieldType, func(args []reflect.Value) []reflect.Value {
			items = reflect.Append(items, args[0])
			return []reflect.Value{reflect.ValueOf(true)}
		})
		rv.Call([]reflect.Value{yieldFunc})
		return items.Interface(), true
	}

	var items []any
	yieldFunc := reflect.MakeFunc(yieldType, func(args []reflect.Value) []reflect.Value {
		items = append(items, []any{args[0].Interface(), args[1].Interface()})
		return []reflect.Value{reflect.ValueOf(true)}
	})
	rv.Call([]reflect.Value{yieldFunc})
	return items, true
}
