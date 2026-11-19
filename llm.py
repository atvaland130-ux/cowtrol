"""LM Studio client (OpenAI-compatible API) for the operator goal and grounded explanations.
 
The LLM never sets fan speed directly. It reads the operator's free text (anything: "milk prices are high",
"power bill is too big this month", "vet visit tomorrow, keep them comfy") and turns it into two numbers the
controller uses: the target for predicted body temperature rise and a minimum fan speed. Both are clamped to the
limits in config.yaml -> goal -> limits, and the hard safety rules still apply afterwards."""
import json
import re
 
import requests
 
from control import clamp_goal
 
 
class LLM:
    def __init__(self, cfg):
        self.host = cfg["llm"]["host"].rstrip("/")
        self.model = cfg["llm"]["model"]
        self.timeout = cfg["llm"]["timeout_s"]
 
    def _chat(self, messages, temperature=0.2, schema=None):
        body = {"model": self.model, "messages": messages, "temperature": temperature}
        if schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "out", "strict": True, "schema": schema}}
        r = requests.post(f"{self.host}/v1/chat/completions", json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
 
    def online(self):
        try:
            requests.get(f"{self.host}/v1/models", timeout=3).raise_for_status()
            return True
        except requests.RequestException:
            return False
 
    def set_goal(self, text, cfg, current, last_decision=None):
        """Free text -> {"target_rise_c", "min_duty_pct", "reason", "clamped"}. Raises ValueError on bad output."""
        lim, dft = cfg["goal"]["limits"], cfg["goal"]["default"]
        schema = {"type": "object", "additionalProperties": False,
                  "required": ["target_rise_c", "min_duty_pct", "reason"],
                  "properties": {"target_rise_c": {"type": "number"}, "min_duty_pct": {"type": "number"},
                                 "reason": {"type": "string"}}}
        now = ""
        if last_decision and last_decision.get("areas"):
            now = "Right now, per zone: " + "; ".join(
                f"{a['area']}: ambient THI {a['thi_ambient']}, predicted rise {a['rise_fan_off_c']} C with fans off, "
                f"{a['rise_fan_max_c']} C with fans at 100%, {a['lying']:.0%} of cows lying, hour {a['hour']}"
                for a in last_decision["areas"]) + "."
        system = f"""You set the cooling goal for an automated dairy barn fan controller from the operator's message.
The controller predicts each cow's body temperature rise above her normal (deg C) with a model trained on real
cows, and runs the fans at the LOWEST speed that keeps that rise under target_rise_c, never below min_duty_pct.
 
Choose:
- target_rise_c, between {lim['target_rise_c'][0]} and {lim['target_rise_c'][1]}. Lower = more cooling.
  Default {dft['target_rise_c']} (a resting cow at THI 68, where heat stress begins).
  Around 0.05-0.1: protect milk yield, welfare, sick/fresh/high-value cows, or cooling matters more than cost.
  Around 0.25-0.4: save energy or money when cows can tolerate it.
- min_duty_pct, between {lim['min_duty_pct'][0]} and {lim['min_duty_pct'][1]}. Normally 0. Raise it only if the
  operator asks for a guaranteed airflow (e.g. "keep the fans on", flies, odour, visitors).
Background: heat stress lowers milk yield and fertility; cooling costs electricity. Weigh the operator's
situation (prices, costs, herd condition, weather plans) and pick numbers that serve it, even if not "optimal".
The current goal is target_rise_c {current['target_rise_c']}, min_duty_pct {current['min_duty_pct']}. Change only
what the message calls for. {now}
Reply with JSON only: {{"target_rise_c": ..., "min_duty_pct": ..., "reason": "<one short sentence>"}}"""
        raw = self._chat([{"role": "system", "content": system}, {"role": "user", "content": text}],
                         temperature=0, schema=schema)
        m = re.search(r"\{.*\}", raw, re.S)
        out = json.loads(m.group(0) if m else raw)
        try:
            t, d = float(out["target_rise_c"]), float(out["min_duty_pct"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"LLM returned an invalid goal: {out!r}")
        goal = clamp_goal(cfg, t, d)                          # structured-action check: limits always hold
        goal["clamped"] = (goal["target_rise_c"], goal["min_duty_pct"]) != (round(t, 3), round(d, 1))
        goal["reason"] = str(out.get("reason", ""))[:300]
        return goal
 
    def explain(self, question, decisions, retrieved, cfg, history=()):
        docs = "\n\n".join(f"[{d['id']}] ({d['source']}) {d['text']}" for d in retrieved) or "(none found)"
        hl, sf = cfg["heat_load"], cfg["safety"]
        # Plain-language snapshot of the latest decision, so even a small model answers "current" questions
        # from fact instead of guessing from the JSON log.
        cur = decisions[-1] if decisions else None
        if cur:
            duties = cur.get("fan_duty_pct") or []
            # per-zone fan state in plain language, so the model answers left/right questions from fact
            zlines = []
            for a in cfg["areas"]:
                zd = [duties[i - 1] for i in a["fans"] if 0 < i <= len(duties)]
                on = "ON" if any(x > 0 for x in zd) else "OFF"
                zlines.append(f"- {a['name']} zone (quadrants {', '.join(a['quadrants'])}): "
                              f"fans {on} at {', '.join(str(round(x)) for x in zd) or '0'} %")
            now = (f"CURRENT STATE (most recent decision, {cur.get('timestamp','')}):\n"
                   + "\n".join(zlines) + "\n"
                   f"- safety status: {cur.get('safety_status','?')}; overrides: {cur.get('overrides') or 'none'}\n"
                   f"- target rise: {cur.get('goal',{}).get('target_rise_c','?')} C\n"
                   "Use these exact per-zone numbers for any question about which fans are on or the current state.")
        else:
            now = "CURRENT STATE: no decisions yet."
        system = f"""You explain an automated dairy barn fan controller to its operator.
Rules: use ONLY the decision log and guideline excerpts below. Cite guidelines as [G#]. Quote numbers exactly
as they appear in the log. If the answer isn't in the data, say so. Be concise (under 150 words).
 
How decisions are made, for each zone (one side of the barn: 3 quadrants, 2 fans that run together):
1. Each quadrant with cows is checked on its own: its sensor gives ambient THI (NRC 1971). A sensor that is not
   answering borrows the average of the working sensors ("sensor_borrowed": true). The zone's fans run at the speed
   its hottest quadrant needs; the numbers shown are for that quadrant ("hottest_quadrant").
2. Effective THI = ambient THI + air_speed_effect + solar_effect.
   - air_speed_effect (negative = cooling): ETIC air speed term (coefficient {hl['air_speed']['coef']}). Air speed
     is the measured air speed in the zone's hottest quadrant with cows ("hottest_quadrant"; the fans run at the speed that quadrant needs) at that fan %.
   - solar_effect: {hl['solar']['thi_per_wm2']} THI per W/m2 of solar radiation (Mader et al. 2006); solar is
     entered by the operator.
3. A model trained on real cows (MmCows: 10 cows, 14 days, measured body temperature) predicts the body
   temperature rise above the cow's normal ("rise_c", deg C) if these conditions held, from effective THI,
   "clustering" (other cows within {cfg['clustering']['neighbor_radius_m']} m), "lying" (share of cows lying;
   in the data lying cows ran 0.1-0.2 C hotter at the same THI because lying traps heat) and "hour" (local time
   of day: body temperature has a daily rhythm, highest in the evening). It predicts the zone's AVERAGE cow.
4. The fans run at the LOWEST speed, not below the goal's min_duty_pct, that keeps rise_c under target_rise_c.
   The goal comes from the operator's message via the LLM ("goal" in the log, with its reason).
   "ml_duty_pct" is that choice; "rise_fan_off_c" / "rise_fan_max_c" show the rise with fans off and at 100%.
   If no speed reaches the target ("target_reached": false) the fans run at 100%.
5. Hard safety rules then override if needed ("fan_duty_pct" is the final value): ambient THI >= {sf['emergency_thi']} -> 100%;
   ambient THI >= {sf['stress_onset_thi']} -> at least {sf['min_air_speed_m_s']} m/s; no valid readings for
   {sf['stale_data_s']} s -> 100%; input faults -> 100%. Overrides are listed in "overrides".
"standing_flag": fewer cows lying than real cows did at that THI, a sign of heat stress.
"air_speed_measured": false means the air speed table for those quadrants is still a placeholder.
 
{now}
 
GUIDELINE EXCERPTS
{docs}
 
RECENT DECISIONS (oldest first)
{json.dumps(decisions, indent=1)}"""
        msgs = [{"role": "system", "content": system}, *history, {"role": "user", "content": question}]
        return self._chat(msgs, temperature=0.3)
 
