# Cowtrol design notes (this controller)

## Body-temperature model
The model predicts the herd's average body-temperature RISE above each cow's cool baseline, in degrees C, from THI (corrected for air speed and solar), clustering, the share of cows lying, and the time of day. Baseline core temp is about 38.6 C; the heat-stress line used on this dataset is 38.8 C, so the onset rise is about 0.2 C. The default target keeps the rise under 0.15 C (about 38.75 C), a buffer below the 38.8 C line.

## Two control laws
- optimizer: run the lowest fan speed whose predicted rise stays under the target rise. Energy-minimal; fans stay low until the cows near the stress line, then rise.
- proportional (default here): set the fan in proportion to the predicted rise. Zero demand at the baseline line, building to a full-cooling target AIR SPEED of 2.0 m/s (about 4.5 mph) at the target-rise line. Each quadrant converts that target breeze to a fan percent using its own measured air-speed map, so a weak-airflow spot runs a higher percent for the same breeze. When the herd is not warm the fans switch fully off; a quadrant that cannot reach 2.0 m/s even at 100 percent runs at 100 percent.

## Operator goal
The LLM turns free text (for example "milk prices are high") into a target rise (0.05 to 0.4 C) and a minimum fan percent. Lower target = cool harder and earlier; higher = save energy. Hard safety rules still override the operator.

Source: Cowtrol design (this project).
