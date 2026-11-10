# Game design overview

Early notes on how the game plays. Nothing here is final; open questions are
collected at the end.

## Premise

The player is a hero who has come to this land to bring back an artifact from
the castle dungeons. The queen has posted a bounty for its return.

## Two renderers

The game uses two renderers, and which one is active tells the player where
they are:

- **The castle (the surface)** uses only the plain PSX renderer. Nothing is
  neural, and the world looks normal.
- **The dungeon** uses the neural renderer. Some things are still drawn with
  the PSX library on top of it:
  - the player's sword (or other weapon) and shield,
  - the HUD and menus,
  - any humans the player meets in the dungeon.

So people always look solid and ordinary, even when the world around them is
dreamlike.

## The castle

The opening area is a castle with an outdoor market and houses around it.

- **People.** The player can talk to townsfolk outside and to people inside the
  castle. They are drawn with the normal PSX renderer.
- **Shops.** The player can buy supplies, weapons, shields and armour.
- **Blacksmith.** Repairs some rare items, such as the monocle.
- **Opening.** When the game starts, the player character talks to themselves
  about why they have come here, which sets up the premise.
- **The queen.** When the player speaks to her, she sends them down to the
  dungeon entrance and warns them that things are really weird down there.
- **Save point.** The castle has one (see [Saving](#saving)).

## The dungeon

- **Coming and going.** The player can go back up to the castle whenever they
  like to restock or talk to people.
- **Shortcuts.** As the game goes on, the player unlocks shortcuts that make the
  trip back to the castle quicker.
- **People.** Some characters live down in the dungeon. Humans are drawn with
  the PSX renderer, like the people in the castle.
- **Chests.** Chests can be opened. They either give the player an item or are
  trapped.
- **Save points.** A few chosen spots in each level have one.

## Affliction and strangeness

The player has an **affliction** value (the name may change) that changes how
the dungeon behaves around them. A level's strangeness is its own and affliction
doesn't touch it: the colourful, psychedelic look is a separate thing.

- Affliction goes from none to full. It creeps up by itself while the player is
  in the dungeon, taking about ten minutes from nothing to full.
- Being attacked and other things in play raise it faster.
- **Getting rid of it.** Returning to the town, by a shortcut or by backtracking,
  resets it to nothing. So does a calming potion.
- As it grows, the world begins to move and then to reach for the player. None
  of this disturbs the neural renderer itself, so the picture stays sharp.
- **The swell** moves the level's walls, floors and ceilings in long, slow
  waves, and bends the creatures with them. It grows as affliction rises.
- **Tentacles.** From 85% affliction, tentacles grow out of the walls and
  ceilings the player is looking at, up to eight at full affliction. One within
  reach turns on the player for 0.7 seconds and then lunges at where they
  were, so the player can step away. A hit costs 4 HP and adds a little
  affliction. They wither when the player leaves them behind, when affliction
  falls, or when a clear seeing potion takes hold.

### Seeing clearly

Some items let the player see the dungeon as it really is, drawn with the
plain PSX renderer instead of the neural one.

- **Clear seeing potion** (name WIP). The player sees the dungeon as it is for
  about ten seconds. The strangeness and the swell drain away while the
  neural renderer is still drawing, then the picture fades to the plain PSX
  renderer. When it runs out, the same happens backwards. A pale bar under the
  MP bar shows the time left. It doesn't change affliction. It can be used from
  the inventory or the hotbar.
- **Monocle.** A lens the player moves around the screen. Only the part of the
  screen inside the lens is drawn with the PSX renderer.
  - **Using it.** Hotbar only: it can't be used from the inventory screen. Put
    it on the hotbar, then hold its number key or d-pad direction. The monocle
    appears for as long as the button is held.
  - **The camera freezes** at its current position and rotation while the
    monocle is up. The look controls move the lens instead: the mouse on
    keyboard and mouse, the right stick on a controller.
  - **Nothing else while it's up.** The player can't move, attack, block or
    cast. All they can do is move the lens around.
  - **Wear.** It wears out with the time it spends up, so short looks make it
    last much longer.
  - **Rarity.** The monocle is rare. The blacksmith in the castle can repair it.

## Dialogue

- There is no recorded voice acting. Characters make strange babbling, speech-like
  sounds while they talk.
- Lines appear in **speech bubbles** that point at whoever is speaking.
- Text appears **one character at a time**. Holding a button makes it come
  in faster.

## Player, inventory and equipment

- The player has an inventory and equipment slots: weapon, shield, helmet,
  armour (body), gloves and two rings.
- The player has health (HP) and mana (MP). Spells use MP.
- **Stats** for now: Attack, Defence and Magic. What is worn adds to them.
- There is no limit on how many kinds of item the player carries, but at most
  99 of any one.
- **Dropping** an item puts it on the floor in front of the player, where it
  can be picked up again by walking over it. For now it is a little cube drawn
  PS1-style over either renderer.

## Movement and camera

- The game is first person.
- The camera **bobs** slightly as the player walks.
- The player can **jump**.
- **Sprint** starts with L3. It stops when the camera turns faster than a set
  speed, so the player can still turn very slowly while sprinting. It
  is for running in a straight line.

## Controls

| Action             | Mouse / keyboard     | Controller |
|--------------------|----------------------|------------|
| Jump               | Space                | X          |
| Sprint             | Shift                | L3 (click left stick) |
| Use door or lever  | E                    | Circle     |
| Raise shield       | Left click           | L2         |
| Attack with weapon | Right click          | R2         |
| Cast selected spell| *TBD*                | L1         |
| Use hotbar item    | 1, 2, 3, 4           | D-pad up / right / down / left |
| Quick-change spell | *TBD*                | Hold Triangle + d-pad |
| Pause menu         | *TBD (probably Esc)* | Start      |

The shield goes on the left button or trigger and the weapon on the right, so
each hand matches a side.

## HUD

```
┌───────────────────────────────────────────────┐
│ ███████ HP (red)                     (compass) │
│ ████ MP (blue)                                 │
│                                                │
│                                                │
│                                                │
│ [1]                                            │
│ [2]                                   [spell]  │
│ [3]                                            │
│ [4]                                            │
└───────────────────────────────────────────────┘
```

- **Top left:** a red health bar with a blue MP bar under it.
- **Top right:** a floating compass.
- **Bottom left:** quick-use items.
  - With mouse and keyboard, a Minecraft-style hotbar used with keys 1–4,
    standing up the left side of the screen.
  - With a controller, the same items set out in a diamond, one for each d-pad
    direction.
- **Bottom right:** the selected spell.

### Quick-change spells

Spells flagged on the [Spells screen](#spells-screen) go into a quick-change
list, so the player can switch spells without pausing.

The list has no size limit. Spells are shown in the order they were added.

1. Hold the quick-change button (Triangle on a PlayStation controller). The
   quick-change list appears above the selected spell icon in the bottom right.
2. While the button is held, the d-pad moves a selection through the list.
   It does not use hotbar items.
3. Let go of the button. The spell under the selection becomes the selected
   spell, and the list hides again.

## Pause menu

Pressing Start opens a menu in the middle of the screen and pauses the game.
It has:

- Resume
- Inventory
- Spells
- Options
- Quit game

### Inventory screen

```
┌──────────────────────────┬──────────────────────────┐
│ [icon] Healing potion  x3│        ┌──────────┐      │
│ [icon] Rope            x1│        │  player  │      │
│ [icon] Torch           x5│        │  model   │      │
│ ...                      │        └──────────┘      │
│                          │ Weapon  [..]             │
│                          │ Shield  [..]             │
│                          │ Armour  [..]             │
│                          │ ...                      │
│                          │──────────────────────────│
│                          │ Stats                    │
│                          │ HP 40/40  MP 12/12  ...  │
└──────────────────────────┴──────────────────────────┘
        items (left)        model, equipment, stats (right)
```

- **Left: items.** A list of everything the player is carrying. Each row has an
  icon, the item's name and how many the player has. An item on the hotbar shows
  its slot number on its icon.
  - Confirming an item opens a modal window on top asking what to do with it.
    The options depend on the item: something usable can be used, put on (or
    moved on, or taken off) the hotbar, or dropped; gear can be equipped or
    dropped.
  - **Hotbar shortcut.** While an item is selected, pressing a hotbar slot's
    own button puts it on that slot: 1–4 on the keyboard. On a controller the
    d-pad also moves through the list, so hold Square and press the slot's
    d-pad direction.
- **Right column**, from top to bottom: the player model, the equipment slots,
  then stats.
- **Player model.** A 3D model of the player character wearing whatever is
  equipped. It needs the same kind of human model as the people in the castle
  and the dungeon.
- **Equipment slots.** The current equipment. Pressing right moves the
  selection from the item list over to the slots, and left moves it back.
  - Clicking or confirming a slot opens a modal window that lists the items the
    player has that can go in that slot.
  - The window shows how each item would change the player's stats. It can
    also take off what is worn.
  - Confirming an item equips it and closes the window.
- **Stats.** The player's current stats. There is no separate Stats screen;
  this is where the player sees them.

### Spells screen

Laid out like the item list. Each row has the spell's icon, its name and its
MP cost.

Spells that are in the quick-change list are marked in the list.

Confirming a spell opens a modal window that describes the spell and has these
options:

- **Set as current.** Makes it the selected spell, shown in the bottom right of
  the HUD and cast with L1.
- **Cast now.** Casts it straight away from the menu.
- **Add to / remove from quick-change.** Puts the spell in the
  [quick-change list](#quick-change-spells) or takes it out.

## Saving

- The player can only save at **save points**: in the castle, and at a few
  chosen places in the levels.
- **Quitting** saves where the player is, and starting the game again puts them
  back there. This is a suspend, not a save the player can go back to later.
- **Dying** makes the player load one of their saves. The suspend from quitting
  is not one of them.

## Open questions

- A final name for "affliction".
- The keyboard key for casting the selected spell.
- The keyboard key for quick-change. How does keyboard and mouse pick from the
  list (mouse wheel? number keys?)
- Does "cast now" from the pause menu cast the spell the moment the menu
  closes? This would allow mid-fight healing without the game running.
- What the artifact is, and why the dungeon is strange.
- What goes in Options. psx-bevy's controls menu rebinds keys the game doesn't
  read through it, and its display menu changes the screen size the networks
  draw at.
