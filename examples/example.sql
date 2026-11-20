-- Example server-side SQL for custom items
-- These entries correspond to examples/example.json

-- item_basic
INSERT INTO item_basic (itemid, name, sortname, stacksize, flags, ah, nosale, basesell)
VALUES (22501, 'iron_blade', 'iron_blade', 1, 63040, 0, 0, 1);

-- item_equipment
INSERT INTO item_equipment (itemid, name, level, ilevel, jobs, MId, shieldSize)
VALUES (22501, 'iron_blade', 1, 0, 4194303, 0, 0);

-- item_weapon
INSERT INTO item_weapon (itemid, name, skill, subskill, dmgType, delay, dmg)
VALUES (22501, 'iron_blade', 3, 0, 3, 240, 50);
