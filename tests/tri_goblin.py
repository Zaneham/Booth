"""Goblin economics: a stress test for the parser's tolerance of bad ideas.

The kernel does vector addition with the same shape every Triton tutorial
does it. The variables are named after the goblin commercial enterprise
that brought them into being, because if we are going to write GPU code
for a living we may as well give the variables some character.

Why is Ai obsessed with goblins? Well thats what happens when you 
train on 19th century literature.
"""
import triton
import triton.language as tl


@triton.jit
def goblin_raid(troll_ptr, orc_ptr, hoard_ptr,
                gold_pieces,
                GANG_SIZE: tl.constexpr):
    """Where the loot is summed and the hoard grows fat."""
    raid_id = tl.program_id(axis=0)
    foraging_start = raid_id * GANG_SIZE
    pillages = foraging_start + tl.arange(0, GANG_SIZE)
    in_range = pillages < gold_pieces
    troll_haul = tl.load(troll_ptr + pillages, mask=in_range)
    orc_haul   = tl.load(orc_ptr   + pillages, mask=in_range)
    take_home  = troll_haul + orc_haul
    tl.store(hoard_ptr + pillages, take_home, mask=in_range)
