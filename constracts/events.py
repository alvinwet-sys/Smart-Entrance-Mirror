# -*- coding: utf-8 -*-
# events.py

from __future__ import annotations
from typing import Dict, Any

FaceIdResolved: Dict[str, Any] = {
"required":["ts","trace_id","source","keyword","confidence"],
"properties":{
"ts":{"type":"number"},
"trace_id":{"type":"string"},
"source":{"type":"string","enum":["voice"]},
"keyword":{"type":"string","minLength":1,"maxLength":32},
"confidence":{"type":"number","minimum":0,"maximum":1},
"channel":{"type":"string","enum":["near","far"],"default":"near"},
"vad":{"type":"boolean","default":True}
},
"additionalProperties":False
}


AsrText: Dict[str, Any] = {
"$schema":"http://json-schema.org/draft-07/schema#",
"title":"AsrText",
"type":"object",
"required":["ts","trace_id","source","text","confidence"],
"properties":{
"ts":{"type":"number"},
"trace_id":{"type":"string"},
"source":{"type":"string","enum":["voice"]},
"text":{"type":"string","minLength":1,"maxLength":256},
"confidence":{"type":"number","minimum":0,"maximum":1},
"lang":{"type":"string","default":"zh-CN"}
},
"additionalProperties":False
}

TtsDone: Dict[str, Any] = {
"$schema":"http://json-schema.org/draft-07/schema#",
"title":"TtsDone",
"type":"object",
"required":["ts","trace_id","source","ok","stopped"],
"properties":{
"ts":{"type":"number"},
"trace_id":{"type":"string"},
"source":{"type":"string","enum":["voice"]},
"ok":{"type":"boolean"},
"stopped":{"type":"boolean"}
},
"additionalProperties":False
}


Wakeup: Dict[str, Any] = {
"$schema":"http://json-schema.org/draft-07/schema#",
"title":"Wakeup",
"type":"object",
"required":["ts","trace_id","source","keyword","confidence"],
"properties":{
"ts":{"type":"number"},
"trace_id":{"type":"string"},
"source":{"type":"string","enum":["voice"]},
"keyword":{"type":"string","minLength":1,"maxLength":32},
"confidence":{"type":"number","minimum":0,"maximum":1},
"channel":{"type":"string","enum":["near","far"],"default":"near"},
"vad":{"type":"boolean","default":True}
},
"additionalProperties":False
}


AsrText: Dict[str, Any] = {
"$schema":"http://json-schema.org/draft-07/schema#",
"title":"AsrText",
"type":"object",
"required":["ts","trace_id","source","text","confidence"],
"properties":{
"ts":{"type":"number"},
"trace_id":{"type":"string"},
"source":{"type":"string","enum":["voice"]},
"text":{"type":"string","minLength":1,"maxLength":256},
"confidence":{"type":"number","minimum":0,"maximum":1},
"lang":{"type":"string","default":"zh-CN"}
},
"additionalProperties":False
}


DecisionRequest: Dict[str, Any] = {
"$schema":"http://json-schema.org/draft-07/schema#",
"title":"DecisionRequest",
"type":"object",
"required":["ts","trace_id","query"],
"properties":{
"ts":{"type":"number"},
"trace_id":{"type":"string"},
"source":{"type":"string","enum":["core"],"default":"core"},
"query":{"type":"string","minLength":1,"maxLength":256},
"context":{
"type":"object",
"required":[],
"properties":{
"identity":{"type":"string"},
"last_asr":{"type":"string","minLength":1,"maxLength":256}
},
"additionalProperties":False
}
},
"additionalProperties":False
}


DecisionReady: Dict[str, Any] = {
"$schema":"http://json-schema.org/draft-07/schema#",
"title":"DecisionReady",
"type":"object",
"required":["ts","trace_id","reply_text"],
"properties":{
"ts":{"type":"number"},
"trace_id":{"type":"string"},
"source":{"type":"string","enum":["llm"],"default":"llm"},
"reply_text":{"type":"string","minLength":1,"maxLength":512},
"actions":{"type":"array","items":{"type":"string"},"default":[]},
"priority":{"type":"integer","minimum":0,"maximum":10,"default":7}
},
"additionalProperties":False
}


ErrorEvent: Dict[str, Any] = {
"$schema":"http://json-schema.org/draft-07/schema#",
"title":"ErrorEvent",
"type":"object",
"required":["ts","trace_id","source","error_code","message"],
"properties":{
"ts":{"type":"number"},
"trace_id":{"type":"string"},
"source":{"type":"string"},
"error_code":{"type":"string"},
"message":{"type":"string"},
"retry_after_ms":{"type":"integer","minimum":0}
},
"additionalProperties":False
}


# 添加缺失的TTS相关schema
TtsCommand: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "TtsCommand",
    "type": "object",
    "required": ["ts", "trace_id", "text"],
    "properties": {
        "ts": {"type": "number"},
        "trace_id": {"type": "string"},
        "source": {"type": "string", "enum": ["core"], "default": "core"},
        "text": {"type": "string", "minLength": 1, "maxLength": 512},
        "priority": {"type": "integer", "minimum": 0, "maximum": 10, "default": 5}
    },
    "additionalProperties": False
}

TtsStop: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "TtsStop",
    "type": "object",
    "required": ["ts", "trace_id"],
    "properties": {
        "ts": {"type": "number"},
        "trace_id": {"type": "string"},
        "source": {"type": "string", "enum": ["core"], "default": "core"}
    },
    "additionalProperties": False
}

SCHEMAS: Dict[str, Dict[str, Any]] = {
    "core.face_id_resolved": FaceIdResolved,
    "core.tts_say": TtsCommand,
    "core.tts_stop": TtsStop,
    "voice.tts_done": TtsDone,
    "voice.wakeup": Wakeup,
    "voice.asr_text": AsrText,
    "core.decision_request": DecisionRequest,
    "llm.decision_ready": DecisionReady,
    "core.error_event": ErrorEvent
}