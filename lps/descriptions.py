"""
FFXI Item Description Formatter

Generates retail-accurate item descriptions from mod ID/value pairs.
Handles stat ordering, quoting conventions, element icons, line wrapping,
and all the quirky formatting rules FFXI uses.

Usage:
    from lps.descriptions import build_description, format_mod

    # Armor: DEF:25 STR+5 "Store TP"+3
    desc = build_description([(1, 25), (8, 5), (73, 3)])

    # Weapon: DMG:99 Delay:224 STR+5
    desc = build_description([(8, 5)], dmg=99, delay=224)
"""

# Max characters per description line (FFXI client limit)
LINE_MAX = 32

# ============================================================
# Mod display names
# ============================================================

MOD_DISPLAY = {
    1: "DEF", 2: "HP", 3: "HP%", 5: "MP", 6: "MP%",
    7: "Convert HP to MP",
    8: "STR", 9: "DEX", 10: "VIT", 11: "AGI", 12: "INT", 13: "MND", 14: "CHR",
    15: "Fire res.", 16: "Ice res.", 17: "Wind res.", 18: "Earth res.",
    19: "Thunder res.", 20: "Water res.", 21: "Light res.", 22: "Dark res.",
    23: "Attack", 24: "Ranged Attack", 25: "Accuracy", 26: "Ranged Accuracy",
    27: "Enmity", 28: "Magic Atk. Bonus", 29: "Magic Defense",
    30: "Magic Accuracy", 31: "Magic Evasion",
    46: "Light Accuracy",
    68: "Evasion", 69: "Ranged Defense", 70: "Ranged Evasion",
    71: "MP recovered while healing",
    73: "Store TP",
    74: None,   # Mana Cede — text-only
    76: None,   # Movement Speed — text-only
    80: "Hand-to-Hand skill", 81: "Dagger skill", 82: "Sword skill", 83: "Great Sword skill",
    84: "Axe skill", 85: "Great Axe skill", 86: "Scythe skill", 87: "Polearm skill",
    88: "Katana skill", 89: "Great Katana skill", 90: "Club skill", 91: "Staff skill",
    97: "Boost Effect",
    104: "Archery skill", 105: "Marksmanship skill", 106: "Throwing skill",
    107: "Guard skill", 108: "Evasion skill", 109: "Shield skill", 110: "Parry skill",
    111: "Divine magic skill", 112: "Healing magic skill", 113: "Enhancing magic skill",
    114: "Enfeebling magic skill", 115: "Elemental magic skill", 116: "Dark magic skill",
    117: "Summoning magic skill", 118: "Ninjutsu skill", 119: "Singing skill",
    120: "String instr. skill", 121: "Wind instr. skill", 122: "Blue magic skill",
    160: "Damage taken", 161: "Phys. dmg. taken",
    162: "Breath dmg. taken", 163: "Magic dmg. taken", 164: "Ranged dmg. taken",
    165: "Crit. hit rate", 166: "Crit. hit evasion",
    167: "Haste", 168: None,  # Spell interruption — handled specially
    170: "Fast Cast", 173: None,  # Martial Arts — text-only
    174: "Skillchain Bonus", 175: "Skillchain Dmg.",
    250: "Slow Resistance",
    256: None, 258: "Mijin Reraise", 259: "Dual Wield",  # 256=Aftermath — text-only
    288: "Double Attack", 289: "Subtle Blow", 290: "Enfeebling Magic potency",
    291: "Counter", 292: "Kick Attacks", 295: "Clear Mind", 296: "Conserve MP",
    302: "Triple Attack", 303: "Treasure Hunter",
    305: "Recycle", 306: "Zanshin", 308: "Ninja Tool Expertise",
    310: 'Enhances "Cursna"',
    311: "Magic Damage", 315: None,  # ENH_DRAIN_ASPIR — text-only
    336: '"Light Arts" skill', 337: '"Dark Arts" skill', 339: "Regen duration",
    345: "TP Bonus", 346: None, 355: None,  # Perpetuation + Weaponskill — handled specially
    357: "Blood Pact Delay", 359: "Rapid Shot",
    364: "Reward HP Bonus", 365: "Snapshot",
    368: "Regain", 369: "Refresh", 370: "Regen",
    372: None,  # Weather dmg. reduction — handled specially
    374: "Cure Potency", 375: "Cure Potency Rcvd.",
    383: "Haste", 384: "Haste",
    386: "Kick Dmg.", 401: "Sublimation bonus", 402: "Wyvern Breath",
    407: "Fast Cast", 414: "Retaliation",
    421: "Crit. hit dmg.", 422: "Double Shot", 423: "Snapshot bonus",
    424: "Ranged Atk. bonus", 426: "Absorb phys. dmg. as MP",
    427: "Enmity Loss Reduction",
    428: '"Perfect Counter" Attack', 429: '"Footwork" Attack bonus',
    430: "Quad Attack", 431: None,  # ITEM_ADDEFFECT_TYPE — internal
    432: "Enspell Dmg. bonus",
    433: '"Minne"', 434: '"Minuet"', 435: '"Paeon"', 436: '"Requiem"',
    437: '"Threnody"', 438: '"Madrigal"', 439: '"Mambo"',
    440: '"Lullaby"', 441: '"Etude"', 442: '"Ballad"', 443: '"March"',
    444: '"Finale"', 445: '"Carol"', 446: '"Mazurka"', 447: '"Elegy"',
    448: '"Prelude"', 449: '"Hymnus"', 450: '"Virelai"',
    451: '"Scherzo"', 452: "All Songs",
    454: None,  # Song duration — text-only
    455: "Song spellcasting time",
    459: "Fire Absorb", 460: "Ice Absorb", 461: "Wind Absorb", 462: "Earth Absorb",
    463: "Thunder Absorb", 464: "Water Absorb", 465: "Light Absorb", 466: "Dark Absorb",
    484: '"Auspice" effect',
    486: "Tactical Parry", 488: "Inhibit TP",
    499: "Additional effect", 500: None,  # ITEM_ADDEFFECT_DMG — internal
    501: "Additional effect",
    508: '"Third Eye" Counter rate',
    519: "Cure cast time", 521: "Absorb",
    522: "Ninjutsu Dmg.", 529: "Enhances Refresh",
    539: "Stoneskin Bonus", 540: None,  # Elemental Siphon — text-only
    544: "Fire elemental bonus", 545: "Ice elemental bonus",
    546: "Wind elemental bonus", 547: "Earth elemental bonus",
    548: "Thunder elemental bonus", 549: "Water elemental bonus",
    550: "Light elemental bonus", 551: "Dark elemental bonus",
    562: "Mag. crit. hit rate", 563: "Mag. crit. hit dmg.",
    568: '"Rapture" bonus', 569: '"Ebullience" bonus',
    570: "WS damage",
    676: "Absorb bonus", 708: "Ninja Tool Expertise", 724: "Zanshin bonus",
    757: "BP Delay Reduction",
    826: "Bounty Shot", 827: "Barspell magic def.",
    830: '"Sneak Attack" DEX bonus', 832: '"Aquaveil" count',
    833: "Song recast delay", 836: "Reverse Flourish", 838: "Regen potency",
    839: "Third Eye Anticipate",
    840: "WS damage (all hits)",
    860: "Cure to MP",
    882: '"Phantom Roll" duration',
    884: "Acc./Collab. Effect",
    890: None,  # Enhancing magic duration — handled specially
    891: "Enhances Coursers Roll", 892: "Enhances Casters Roll",
    893: "Enhances Blitzers Roll", 894: "Enhances Allies Roll",
    895: "Enhances Tacticians Roll", 896: "Despoil",
    898: "Smite", 899: "Tactical Guard", 900: "Utsusemi Bonus",
    901: "Elemental Celerity", 902: "Occult Acumen",
    903: "Fencer TP Bonus", 904: "Fencer Crit Rate",
    905: "Shield Def. Bonus", 908: "Critical Defense",
    910: "Divine Benison", 912: "Conspirator", 913: "Blood Boon",
    951: "Additional effect", 964: "Ranged crit. dmg.",
    1038: "Double Attack dmg.", 1039: "Triple Attack dmg.",
    1040: "Avatar: Level bonus",
    1045: "Restraint", 1046: "Blood Rage", 1047: "Counter Damage",
    1090: "Sengikori", 1097: "Impetus",
    1139: "Burst Affinity", 1140: "Chain Affinity",
    1146: "Elem. magic recast",
    1147: "Blue Magic recast", 1148: "Enhances Futae",
    1153: "Absorb TP", 1154: "Avatar's Favor", 1157: "Enhances Spur",
    1170: "Mana Cede", 1171: "Mana Cede II",
    1194: None,  # DAY_WEATHER_PROC_BONUS — text-only
}

# DEF uses colon format (DEF:25)
COLON_MODS = {1}

# Stats wrapped in quotes: "Store TP"+3
QUOTED_MODS = {
    73, 165, 170, 173, 259, 288, 289, 291, 292, 295, 296, 302, 303, 305,
    306, 308, 345, 346, 357, 359, 365, 368, 369, 370, 374, 375, 386, 407,
    421, 427, 430, 486, 488, 519, 562, 563, 570, 840, 908, 964,
}

# Elemental resistance icons (FFXI Private Use Area)
ELEMENT_ICONS = {
    15: "\uEF1F", 16: "\uEF20", 17: "\uEF21", 18: "\uEF22",
    19: "\uEF23", 20: "\uEF24", 21: "\uEF25", 22: "\uEF26",
}

# Internal mods — never shown in descriptions
HIDDEN_MODS = {276, 431, 500}

# Mods shown as text phrases (no numeric value displayed)
TEXT_ONLY_MODS = {
    74:   'Enhances "Mana Cede" effect',
    76:   "Increases movement speed",
    173:  'Enhances "Martial Arts" effect',
    256:  "Aftermath",
    310:  'Enhances "Cursna" effect',
    315:  'Enhances "Drain" and "Aspir"',
    355:  "Grants weaponskill",
    414:  'Augments "Retaliation"',
    454:  "Increases song effect duration",
    529:  'Enhances "Refresh" effect',
    540:  'Enhances "Elemental Siphon" effect',
    912:  'Augments "Conspirator"',
    1045: 'Augments "Restraint"',
    1046: 'Augments "Blood Rage"',
    1097: 'Augments "Impetus"',
    1139: 'Augments "Burst Affinity"',
    1140: 'Augments "Chain Affinity"',
    1148: 'Enhances "Futae" effect',
    1157: 'Enhances "Spur" effect',
    1170: 'Enhances "Mana Cede" effect',
    1171: 'Enhances "Mana Cede" effect',
    1194: 'Weather/day: Enhances bonus',
}

# Stored as value * 100, displayed as percentage
SCALED_PERCENT_MODS = {
    160, 161, 162, 163, 164, 190,  # Damage taken
    384, 383, 167,                  # Haste
    421, 562, 563, 964,             # Crit damage
    175,                            # Skillchain damage
    1146,                           # Elemental magic recast
}

# Value IS the percentage (no scaling)
RAW_PERCENT_MODS = {
    165, 170, 407,                  # Crit rate, Fast Cast
    288, 302, 430,                  # DA, TA, QA
    374, 260, 375,                  # Cure Potency
    305, 359, 365, 422, 519,        # Recycle, Rapid Shot, Snapshot, Double Shot, Cure cast
}

# Display priority (lower = shown first)
STAT_PRIORITY = {
    1: 1,
    2: 30, 5: 31,
    8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16,
    23: 20, 24: 21, 25: 22, 26: 23,
    28: 24, 30: 25,
    68: 2, 163: 3, 31: 4,
    384: 35, 383: 35, 167: 36,
    165: 40, 421: 41,
    27: 45, 73: 46, 289: 47,
    288: 50, 302: 51, 430: 52,
    369: 53, 370: 54, 368: 55,
    311: 56, 170: 57, 407: 57,
    374: 58, 259: 59, 346: 60, 357: 61,
}


def format_mod(mod_id: int, value: int) -> str | None:
    """
    Format a single mod for display in an item description.
    Returns None if the mod should be hidden.
    """
    if mod_id in HIDDEN_MODS:
        return None

    if mod_id in TEXT_ONLY_MODS:
        return TEXT_ONLY_MODS[mod_id]

    # Mods stored positive but displayed as negative reduction
    if mod_id == 346:
        return f"Avatar perpetuation cost -{value}"
    if mod_id == 833:
        return f"Song recast delay -{value}"
    if mod_id == 372:
        return f"Weather: Dmg. taken -{value}"
    if mod_id == 455:
        return f"Song spellcasting time -{value}%"
    if mod_id == 168:
        return f"Spell interruption rate down {value}%"
    if mod_id == 890:
        return f"Enhancing magic duration +{value}%"

    # Elemental resistances use icon characters
    if mod_id in ELEMENT_ICONS:
        sign = "+" if value > 0 else ""
        return f"{ELEMENT_ICONS[mod_id]}{sign}{value}"

    name = MOD_DISPLAY.get(mod_id)
    if name is None:
        return f"Mod#{mod_id}+{value}"

    if mod_id in COLON_MODS:
        return f"{name}:{value}"

    # Don't double-quote names that already contain quotes
    quoted = mod_id in QUOTED_MODS and '"' not in name
    display_name = f'"{name}"' if quoted else name

    if mod_id in SCALED_PERCENT_MODS:
        pct = round(value / 100)
        sign = "+" if pct > 0 else ""
        return f"{display_name}{sign}{pct}%"

    if mod_id in RAW_PERCENT_MODS:
        sign = "+" if value > 0 else ""
        return f"{display_name}{sign}{value}%"

    sign = "+" if value > 0 else ""
    return f"{display_name}{sign}{value}"


def get_priority(mod_id: int) -> int:
    """Get display sort priority for a mod (lower = shown first)."""
    return STAT_PRIORITY.get(mod_id, 999)


def build_description(mods: list, dmg: int = None, delay: int = None,
                      line_max: int = LINE_MAX) -> str:
    """
    Build an item description string with proper line wrapping.

    Args:
        mods: List of (mod_id, value) tuples
        dmg: Weapon damage (None for armor)
        delay: Weapon delay (None for armor)
        line_max: Maximum characters per line (default 32)

    Returns:
        Description string with \\n line breaks.
    """
    parts = []

    if dmg is not None and delay is not None:
        parts.append(f"DMG:{dmg} Delay:{delay}")

    sorted_mods = sorted(mods, key=lambda m: get_priority(m[0]))

    for mod_id, value in sorted_mods:
        if value == 0:
            continue
        formatted = format_mod(mod_id, value)
        if formatted is not None:
            parts.append(formatted)

    if not parts:
        return ""

    lines = []
    current = ""

    for part in parts:
        if not current:
            current = part
        elif len(current) + 1 + len(part) <= line_max:
            current += " " + part
        else:
            lines.append(current)
            current = part

    if current:
        lines.append(current)

    return "\\n".join(lines)
