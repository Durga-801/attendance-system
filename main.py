import argparse
import json
import os
from pathlib import Path

DATA_FILE = Path(__file__).parent / "students.json"
MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
MARKS_RULE = [(90, 5), (80, 4), (75, 3), (65, 2), (50, 1)]  # (min %, marks)


def load_data():
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def save_data(students):
    DATA_FILE.write_text(json.dumps(students, indent=2), encoding="utf-8")


def find_student(students, roll_no):
    for s in students:
        if s["roll_no"] == str(roll_no):
            return s
    raise ValueError(f"Roll no {roll_no} not found")


def calculate_attendance_percent(roll_no):
    s = find_student(load_data(), roll_no)
    pct = round(s["days_present"] / s["total_days"] * 100, 2)
    return {"roll_no": s["roll_no"], "name": s["name"], "attendance_percent": pct}


def calculate_marks(roll_no):
    pct = calculate_attendance_percent(roll_no)["attendance_percent"]
    marks = next((m for limit, m in MARKS_RULE if pct >= limit), 0)
    return {"roll_no": str(roll_no), "attendance_percent": pct, "marks": marks}


def update_attendance(roll_no, days_present=None, total_days=None):
    students = load_data()
    s = find_student(students, roll_no)
    if total_days is not None:
        s["total_days"] = int(total_days)
    if days_present is not None:
        s["days_present"] = int(days_present)
    if s["days_present"] > s["total_days"]:
        raise ValueError("days_present cannot be greater than total_days")
    save_data(students)
    return {"updated": s}


def list_students():
    return load_data()


TOOL_FUNCS = {
    "calculate_attendance_percent": calculate_attendance_percent,
    "calculate_marks": calculate_marks,
    "update_attendance": update_attendance,
    "list_students": list_students,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "list_students",
        "description": "Get all students with name, roll_no, total_days, days_present.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "calculate_attendance_percent",
        "description": "Attendance percentage of one student.",
        "parameters": {"type": "object",
                       "properties": {"roll_no": {"type": "string"}},
                       "required": ["roll_no"]}}},
    {"type": "function", "function": {
        "name": "calculate_marks",
        "description": "Attendance percentage and attendance marks of one student.",
        "parameters": {"type": "object",
                       "properties": {"roll_no": {"type": "string"}},
                       "required": ["roll_no"]}}},
    {"type": "function", "function": {
        "name": "update_attendance",
        "description": "Update days_present and/or total_days of a student.",
        "parameters": {"type": "object",
                       "properties": {"roll_no": {"type": "string"},
                                      "days_present": {"type": "integer"},
                                      "total_days": {"type": "integer"}},
                       "required": ["roll_no"]}}},
]

SYSTEM_PROMPT = (
    "You are an attendance assistant. Use the provided tools to get real data; "
    "never guess numbers. Final answer must be ONLY valid JSON, no extra text."
)


def _args(tool_call):
    a = tool_call.function.arguments
    return json.loads(a) if isinstance(a, str) else a


def run_ai(user_prompt):
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=MODEL, token=os.environ["HF_TOKEN"])
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]
    for _ in range(10):
        msg = client.chat_completion(messages=messages, tools=TOOLS,
                                     max_tokens=1500).choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": t.id, "type": "function",
                                         "function": {"name": t.function.name,
                                                      "arguments": json.dumps(_args(t))}}
                                        for t in msg.tool_calls]})
        for t in msg.tool_calls:
            try:
                result = TOOL_FUNCS[t.function.name](**_args(t))
            except Exception as e:
                result = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": t.id,
                             "name": t.function.name, "content": json.dumps(result)})
    raise RuntimeError("AI did not finish in time")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", default="Calculate attendance percent and marks for all students.")
    p.add_argument("--offline", action="store_true", help="run without AI")
    p.add_argument("--update", metavar="ROLL_NO")
    p.add_argument("--present", type=int)
    p.add_argument("--total", type=int)
    a = p.parse_args()

    if a.update:
        out = update_attendance(a.update, a.present, a.total)
    elif a.offline:
        out = {"students": [{**calculate_marks(s["roll_no"]), "name": s["name"]}
                            for s in load_data()]}
    else:
        text = run_ai(a.prompt)
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            out = {"raw_output": text}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()