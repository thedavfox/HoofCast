# Horse behavior prediction data dictionary

Each row is one contiguous 200-second observation. The first 100 video seconds
form the pre-period covariates. The final 100 seconds form the outcome period.
Blocks do not cross a 60-second recording gap or a calendar-month boundary.

## Identifiers and timing

| Variable | Meaning |
| --- | --- |
| `observation_id` | Unique observation identifier. |
| `block_start_timestamp_local` | Local timestamp at the beginning of the 200-second video block. |
| `time_of_day_hour` | Local hour of the block start, coded 0 through 23. For example, 14 means 2:00 PM through 2:59 PM. |
| `recording_month` | Calendar month of the block start, in `YYYY-MM` form. |
| `block_video_seconds` | Total video duration in the observation. Always 200 seconds. |
| `pre_video_seconds` | Video duration in the covariate period. Always 100 seconds. |
| `outcome_video_seconds` | Video duration in the outcome period. Always 100 seconds. |

## Pre-period video covariates

All variables below describe only the first 100 video seconds. They are
duration-weighted proportions from 0 to 1. For example,
`pre_posture_lying_lateral = 0.30` means the horse was labeled lying laterally
for 30 of the 100 pre-period seconds.

### Posture

| Variable | Meaning |
| --- | --- |
| `pre_posture_standing` | Proportion of pre-period seconds labeled standing. |
| `pre_posture_lying_sternal` | Proportion labeled lying sternally. |
| `pre_posture_lying_lateral` | Proportion labeled lying laterally. |
| `pre_posture_transitioning` | Proportion labeled transitioning between postures. |
| `pre_posture_unknown` | Proportion with an unknown posture label. |

The five posture proportions sum to 1 for each observation.

### Visibility

| Variable | Meaning |
| --- | --- |
| `pre_visibility_full` | Proportion of pre-period seconds when the horse was fully visible. |
| `pre_visibility_partial` | Proportion when the horse was partially visible. |
| `pre_visibility_not_visible` | Proportion when the horse was not visible. |

The three visibility proportions sum to 1 for each observation.

### Posture transition

| Variable | Meaning |
| --- | --- |
| `pre_transition_stand_to_lie` | Proportion of pre-period seconds labeled as a stand-to-lie transition. |
| `pre_transition_lie_to_stand` | Proportion labeled as a lie-to-stand transition. |
| `pre_transition_none` | Proportion labeled with no transition. |
| `pre_transition_unknown` | Proportion with an unknown transition label. |

The four transition proportions sum to 1 for each observation.

### Secondary behavior

| Variable | Meaning |
| --- | --- |
| `pre_pawing` | Proportion of pre-period seconds with pawing observed. |
| `pre_biting_object` | Proportion with biting an object observed. |
| `pre_rubbing` | Proportion with rubbing observed. |
| `pre_repetitive_movement` | Proportion with repetitive movement observed. |
| `pre_no_secondary_behavior` | Proportion with none of the listed secondary behaviors observed. |

Secondary behaviors are separate indicators and may overlap, so this group is
not required to sum to 1.

## Outcome variables

All outcome variables describe only the final 100 video seconds. The six
mutually exclusive activity proportions are duration-weighted values from 0 to
1 and sum to 1 for each observation. `outcome_activity_not_stationary` is a
derived complement rather than an additional mutually exclusive category.

| Variable | Meaning |
| --- | --- |
| `outcome_activity_eating` | Proportion of outcome seconds labeled eating. |
| `outcome_activity_drinking` | Proportion labeled drinking. |
| `outcome_activity_walking` | Proportion labeled walking. |
| `outcome_activity_stationary` | Proportion labeled stationary. |
| `outcome_activity_not_stationary` | Complement of the stationary proportion: `1 - outcome_activity_stationary`. This is the primary prediction target. It includes walking, eating, drinking, other activity, and unknown activity. |
| `outcome_activity_other` | Proportion labeled another activity. |
| `outcome_activity_unknown` | Proportion with an unknown activity label. |

For example, `outcome_activity_not_stationary = 0.42` means that the horse was
not labeled stationary for 42 of the final 100 video seconds.

## Weather covariates

Weather values come from the ERA5 historical reanalysis at Bainbridge Island,
Washington. They are matched to the local hourly interval containing the block
start. They describe gridded historical weather conditions, not measurements
from an instrument at the barn.

| Variable | Meaning |
| --- | --- |
| `weather_hour_start_local` | Start of the local hourly weather interval used for the observation. |
| `weather_temperature_c` | Air temperature at 2 meters, in degrees Celsius. |
| `weather_relative_humidity_pct` | Relative humidity at 2 meters, as a percent. |
| `weather_precipitation_mm_prior_hour` | Total precipitation, in millimeters, over the preceding hour. |
| `weather_cloud_cover_pct` | Total cloud-cover percentage. |
| `weather_shortwave_radiation_w_m2_prior_hour` | Mean global horizontal shortwave solar radiation over the preceding hour, in watts per square meter. This is the sun-level measure. |
| `weather_wind_speed_10m_kmh` | Wind speed at 10 meters, in kilometers per hour. |
| `weather_wind_direction_10m_degrees` | Wind direction at 10 meters, in degrees. |
| `weather_wind_gusts_10m_kmh` | Wind-gust speed at 10 meters, in kilometers per hour. |
| `weather_is_day` | Day/night flag for the matched weather hour: 1 for day and 0 for night. |
