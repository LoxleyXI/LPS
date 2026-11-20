# Strings Module

Patches DMSG string resources: spell names, zone names, ability names, and 30+ other text resources used by the FFXI client.

## Supported Resources

Spell/ability names, zone names, job names, race names, quest/mission names, key item names, status effects, titles, emotes, weather, days, crafts, regions, nations, menu text, system messages, merits, job points, trusts, mounts, and more.

## Config Format

```json
{
  "custom": {
    "strings": {
      "spell_names": {
        "100": "Custom Fire"
      },
      "zone_names": {
        "400": "Ice Cave"
      }
    }
  }
}
```

## Resources

The names a config's `"strings"` can use:

`ability_descriptions`, `ability_names`, `chat_filter_names`, `craft_names`, `craft_rank_names`, `day_names`, `emote_names`, `job_names`, `job_names_short`, `job_point_category_names`, `job_point_names`, `key_item_descriptions`, `key_item_names`, `menu_text`, `merit_category_names`, `merit_names`, `mission_names`, `monstrosity_ability_names`, `monstrosity_species_names`, `moon_phase_names`, `mount_names`, `nation_names`, `quest_names`, `race_names`, `region_names`, `spell_descriptions`, `spell_names`, `status_descriptions`, `status_names`, `system_messages`, `title_names`, `trust_names`, `unity_names`, `weather_names`, `zone_names`, `zone_names_short`

Known issue: the client file each resource points at is worked out from its file id in a way that
doesn't match how the client finds its files, so most resources point at the wrong file or a missing
one. Check a resource's file against your client before relying on it.
