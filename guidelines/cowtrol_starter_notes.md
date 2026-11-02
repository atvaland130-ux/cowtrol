# Cowtrol starter guideline notes

Summaries written for this project, not copied text. Add extension PDFs to this folder for more detail.

## Heat stress onset (THI 68)
Dairy cows can begin to experience heat stress once the temperature-humidity index (THI) reaches about 68. In humid conditions this can happen at air temperatures in the low 70s F, and high-producing cows in well-ventilated barns can be affected at temperatures in the mid 60s F.
Sources: Penn State Extension, Heat Stress Abatement Techniques for Dairy Cattle (https://extension.psu.edu/heat-stress-abatement-techniques-for-dairy-cattle); UMN Extension, Heat stress in dairy cattle (https://extension.umn.edu/dairy-milking-cows/heat-stress-dairy-cattle)

## THI stress categories (UMN)
UMN Extension groups heat stress by THI: below 68 is no stress; 68 to 71 is mild; 72 to 79 is mild to moderate; 80 to 90 is moderate to severe; 90 to 99 is severe. Respiration rate and body temperature rise with each category, from roughly 40 to 60 breaths per minute with no stress to over 100 breaths per minute when severe.
Source: UMN Extension (https://extension.umn.edu/dairy-milking-cows/heat-stress-dairy-cattle)

## Target air speed over cows
Penn State recommends air speeds of roughly 3.5 to 5 mph over resting, feeding and holding areas to increase convective and evaporative heat loss. University of Kentucky guidance targets about 7.2 ft/s over cows, which is about 4.9 mph.
Sources: Penn State Extension (https://extension.psu.edu/heat-stress-abatement-techniques-for-dairy-cattle); University of Kentucky, Fans: how cool are your cows (https://afs.mgcafe.uky.edu/files/fans_how_cool_are_your_cows.pdf)

## Effect of higher air speed
UMN Extension gives an example at THI 75 where raising air speed from about 3 mph to 10 mph lowered a cow's respiration rate from 68 to 57 breaths per minute, moving it from mild stress to normal.
Source: UMN Extension (https://extension.umn.edu/dairy-milking-cows/heat-stress-dairy-cattle)

## When fans are not enough
Air movement alone helps at the start of heat stress, but once cows reach more severe heat stress, convective cooling may not be sufficient and evaporative cooling (soaking cows with sprinklers plus fans) is recommended.
Source: University of Kentucky (https://afs.mgcafe.uky.edu/files/fans_how_cool_are_your_cows.pdf)

## Fan turn-on temperature
University of Kentucky guidance suggests thermostat-controlled fans that turn on when the temperature at cow level reaches about 73 F.
Source: University of Kentucky (https://afs.mgcafe.uky.edu/files/fans_how_cool_are_your_cows.pdf)

## Air exchange
In summer, barns should exchange their air about once per minute or faster; UMN cites 60 to 90 air exchanges per hour in hot weather.
Sources: Penn State Extension; UMN Extension

## Emergency signs
UMN Extension advises immediate action when more than about 10% of cows breathe faster than 100 breaths per minute, or when more than 5 to 10% have body temperatures above 105 F.
Source: UMN Extension (https://extension.umn.edu/dairy-milking-cows/heat-stress-dairy-cattle)

## Safe agentic control (design principle)
For LLM-assisted barn control, animal welfare is treated as a non-negotiable constraint. The LLM interprets operator intent and explains decisions, while decision-critical checks stay deterministic; hard safety rules can override the operator's objective (for example, keeping fans at maximum under emergency THI even when asked to save energy). Decisions should be logged for audit, and explanations should be grounded in sensor data and cited guidelines.
Source: Choi et al. (2026), LLM-driven agentic digital twins for dairy housing, J. ASABE 69(4): 663-669 (https://doi.org/10.13031/ja.16663)

## Air speed cooling used by this controller (ETIC)
The controller treats moving air as lowering the temperature the cow feels by 0.1173 x v^0.707 x (39.2 - T) degrees C, where v is air speed in m/s and T is air temperature in C. This is the air speed term of the Equivalent Temperature Index for dairy Cattle (ETIC). At 28 C it is worth about 1.8 THI points at 1 m/s and 3.0 THI points at 2 m/s. The cooling shrinks as air temperature approaches cow body temperature (39.2 C).
Source: Wang et al. (2018), A predictive model of equivalent temperature index for dairy cattle (ETIC), J. Thermal Biology (https://www.sciencedirect.com/science/article/abs/pii/S0306456518301050). For comparison, Mader et al. (2006) subtract 1.992 THI points per m/s of wind for feedlot cattle (https://digitalcommons.unl.edu/animalscifacpub/608).

## Clustering (cows close together)
Clustering is measured as the average number of other cows within about one cow length (2.5 m). The controller assumes each nearby cow's body heat (about 500 W of sensible heat) warms the air reaching its neighbours. The warming is n x 500 / (1.2 x 1005 x 3 x v) degrees C for n nearby cows and air speed v (never taken below 0.2 m/s). That is roughly 1 THI point per nearby cow in still air and 0.2 THI points at 1 m/s, so moving air matters more when cows are bunched. This is an engineering estimate, not a measured relationship.
Source for cow heat output: CIGR (2002), Heat and moisture production at animal and house levels (https://www.cigr.org/sites/default/files/documets/CIGR_4TH_WORK_GR.pdf).

## What real cow data showed about clustering and lying (MmCows)
In the public MmCows dataset (10 cows with position tags and body temperature loggers, 14 days), cows with more neighbours had slightly higher body temperature, but the effect disappeared once lying in the stalls was accounted for. An hour of lying went with about 0.45 C higher body temperature than an hour of standing. The data support a clustering effect of at most about 1 THI point per nearby cow, and possibly none. Herd body temperature tracked the past several hours of THI far better than the current reading.
Source: Vu et al. (2024), MmCows: A Multimodal Dataset for Dairy Cattle Monitoring (https://github.com/neis-lab/mmcows); analysis by the Cowtrol team.

## Heat stress score and fan choice
The 0-100 heat stress score is 25 times the expected stress level from the model (0 no stress, 25 mild, 50 moderate, 75 severe, 100 extreme). The fan runs at the lowest speed that brings the score under the operator's target: under 12 for balanced, under 5 for welfare first, under 30 for save energy. If no fan speed reaches the target the fan runs at 100 %.
Source: Cowtrol design decision.
