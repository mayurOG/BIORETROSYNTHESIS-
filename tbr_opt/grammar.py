"""SMILES grammar constraint for seq2seq decoding.

Masks logits at every decode step so the model can only emit syntactically valid
SMILES: balanced parentheses and brackets, ring bonds spanning at least two distinct
chain atoms, legal atom/bond alternation, and well-formed chirality descriptors.

Two things are deliberately out of scope. Valence and aromaticity perception are not
decidable from a token stream, and neither is a ring bond that duplicates an existing
ring path ("n23cnc32"). RDKit stays the final authority for both - see
engine._split_valid, which drops whatever survives decoding but will not parse.
"""

CAT_START, CAT_ATOM, CAT_BOND, CAT_OPEN, CAT_CLOSE, CAT_RING, CAT_DOT = range(7)

# A branch always opens on an atom, never on a bond: "C-(C)" and "O=(C" are illegal.
_NEXT = {
    CAT_START: frozenset({CAT_ATOM}),
    CAT_ATOM: frozenset({CAT_ATOM, CAT_BOND, CAT_OPEN, CAT_CLOSE, CAT_RING, CAT_DOT}),
    CAT_BOND: frozenset({CAT_ATOM, CAT_RING}),
    CAT_OPEN: frozenset({CAT_ATOM, CAT_BOND}),
    CAT_CLOSE: frozenset({CAT_ATOM, CAT_BOND, CAT_OPEN, CAT_CLOSE, CAT_RING, CAT_DOT}),
    CAT_RING: frozenset({CAT_ATOM, CAT_BOND, CAT_OPEN, CAT_CLOSE, CAT_RING, CAT_DOT}),
    CAT_DOT: frozenset({CAT_ATOM}),
}

_TERMINAL_OK = frozenset({CAT_ATOM, CAT_CLOSE, CAT_RING})

BONDS = set("-=#:~/\\")
ORGANIC = set("BCNOPSFI")
AROMATIC = set("bcnops")
_TWO_CHAR_ORGANIC = ("Cl", "Br")
_TWO_CHAR_AROMATIC = ("se", "as")

ELEMENTS = frozenset("""
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn
Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La
Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po
At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg
Cn Nh Fl Mc Lv Ts Og D T
""".split())

AROMATIC_ELEMENTS = frozenset("bcnops")
AROMATIC_ELEMENTS_TWO = ("se", "as")

# Bracket-atom parser stages.
_B_START = 1       # after '[', isotope digits or element symbol
_B_ELEM = 2        # element consumed
_B_AT1 = 3         # saw one '@'
_B_AT2 = 4         # saw '@@'
_B_HCOUNT = 5      # saw 'H'
_B_CHARGE = 6      # saw '+'/'-'
_B_CLASSSIGN = 7   # saw ':', expecting '-' or a digit
_B_CLASS = 8       # atom-class digits
_B_CHIRAL1 = 9     # first chirality-name letter
_B_CHIRAL2 = 12    # second chirality-name letter
_B_CHIRALIDX = 10  # chirality index digits
_B_CLOSEABLE = frozenset({_B_ELEM, _B_AT1, _B_AT2, _B_HCOUNT, _B_CHARGE,
                          _B_CLASS, _B_CHIRAL1, _B_CHIRAL2, _B_CHIRALIDX})
_B_AFTER_ELEM = frozenset({_B_ELEM, _B_AT1, _B_AT2, _B_HCOUNT, _B_CHARGE,
                           _B_CHIRAL1, _B_CHIRAL2, _B_CHIRALIDX})

# The only chirality descriptors RDKit accepts are TH, TB, SP, AL and OH, so the
# second letter is a function of the first. Without this "[C@SH]" parses as chirality.
_CHIRAL_SECOND = {"T": frozenset("HB"), "S": frozenset("P"),
                  "A": frozenset("L"), "O": frozenset("H")}

# A ring bond needs two distinct atoms on the chain that opened it, so each open ring
# tracks atoms emitted since it opened, capped because anything >= 2 behaves the same.
_RING_MATURE = 2

INITIAL = (0, CAT_START, frozenset(), 0, "", "")


def base_key(state):
    """The part of a state that determines legality of every non-ring-bearing token.

    Ring contents are excluded because only tokens containing a digit or '%' can depend
    on them; SmilesGrammarConstraint rechecks exactly that subset instead of rebuilding
    the whole vocabulary mask for every ring configuration.
    """
    depth, last, _rings, stage, pct, chir = state
    return (depth == 0, last, stage, pct, chir)


def is_terminal(state):
    depth, last, rings, stage, pct, chir = state
    return (
        depth == 0
        and not rings
        and stage == 0
        and not pct
        and not chir
        and last in _TERMINAL_OK
    )


def _element(text, i):
    """Match an element symbol at text[i] inside a bracket atom.

    Case matters: [Se] is selenium, [se] is aromatic selenium. Returns (symbol, next_i).
    """
    if i >= len(text):
        return None
    c = text[i]
    if c == "*":
        return "*", i + 1
    two = text[i : i + 2]
    if c.isupper():
        if two in ELEMENTS:
            return two, i + 2
        if c in ELEMENTS:
            return c, i + 1
        return None
    if c.islower():
        if two in AROMATIC_ELEMENTS_TWO:
            return two, i + 2
        if c in AROMATIC_ELEMENTS:
            return c, i + 1
    return None


def _age_rings(rings, depth):
    """Age only rings opened at this paren depth.

    Atoms inside a branch are not on the ring path: in C1(O)CCC1 the branch oxygen must
    not count toward closing ring 1, and in OCC1(O)[C@@H]1O the ring is genuinely too
    small once branch atoms are excluded.
    """
    if not rings:
        return rings
    return frozenset(
        (rid, min(cnt + 1, _RING_MATURE), od) if od == depth else (rid, cnt, od)
        for rid, cnt, od in rings
    )


def _toggle_ring(rings, rid, depth):
    """Open or close a ring bond. Returns None if the closure would be degenerate."""
    for entry in rings:
        if entry[0] == rid:
            cnt, od = entry[1], entry[2]
            # Only decidable when the closure sits at the depth that opened the ring;
            # cross-depth closures like C1CC(C1)C are left to RDKit.
            if od == depth and cnt < _RING_MATURE:
                return None
            return rings - {entry}
    return rings | {(rid, 0, depth)}


def advance(state, text):
    """Feed `text` through the grammar. Returns the new state, or None if illegal.

    Resumable across arbitrary chunk boundaries, which is what lets it run per decode
    step on tokens that split a bracket atom (e.g. "[", "C@@", "H", "]").
    """
    if state is None:
        return None

    depth, last, rings, stage, pct, chir = state
    i, n = 0, len(text)

    while i < n:
        c = text[i]

        if c.isspace():
            i += 1
            continue

        if stage:
            if c == "]":
                if stage not in _B_CLOSEABLE:
                    return None
                rings = _age_rings(rings, depth)
                stage, last, chir, i = 0, CAT_ATOM, "", i + 1
                continue

            if stage == _B_START:
                if c.isdigit():
                    i += 1
                    continue
                if not (c.isalpha() or c == "*"):
                    return None
                match = _element(text, i)
                if match is None:
                    return None
                stage, i = _B_ELEM, match[1]
                continue

            if c == "@":
                if stage == _B_ELEM:
                    stage, i = _B_AT1, i + 1
                    continue
                if stage == _B_AT1:
                    stage, i = _B_AT2, i + 1
                    continue
                return None

            if c == "H" and stage in (_B_ELEM, _B_AT1, _B_AT2, _B_CHIRAL2, _B_CHIRALIDX):
                stage, i = _B_HCOUNT, i + 1
                continue

            if c.isupper() and stage == _B_AT1 and c in _CHIRAL_SECOND:
                chir, stage, i = c, _B_CHIRAL1, i + 1
                continue

            if c.isupper() and stage == _B_CHIRAL1 and c in _CHIRAL_SECOND.get(chir, ()):
                stage, chir, i = _B_CHIRAL2, "", i + 1
                continue

            if c.isdigit() and stage in (_B_CHIRAL1, _B_CHIRAL2, _B_CHIRALIDX):
                stage, chir, i = _B_CHIRALIDX, "", i + 1
                continue

            if c.isdigit() and stage in (_B_HCOUNT, _B_CHARGE, _B_CLASS):
                i += 1
                continue

            if c in "+-" and stage in _B_AFTER_ELEM:
                stage, chir, i = _B_CHARGE, "", i + 1
                continue

            if c == ":" and stage in _B_AFTER_ELEM:
                stage, chir, i = _B_CLASSSIGN, "", i + 1
                continue

            if stage == _B_CLASSSIGN and (c == "-" or c.isdigit()):
                stage, i = _B_CLASS, i + 1
                continue

            return None

        allowed = _NEXT[last]

        if pct:
            if not c.isdigit():
                return None
            if len(pct) == 1:
                pct += c
                i += 1
                continue
            rings = _toggle_ring(rings, int(pct[1] + c), depth)
            if rings is None:
                return None
            pct, last, i = "", CAT_RING, i + 1
            continue

        if c == "%":
            if CAT_RING not in allowed:
                return None
            pct, i = "%", i + 1
            continue

        if c.isdigit():
            if CAT_RING not in allowed:
                return None
            rings = _toggle_ring(rings, int(c), depth)
            if rings is None:
                return None
            last, i = CAT_RING, i + 1
            continue

        if c == "(":
            if CAT_OPEN not in allowed:
                return None
            depth, last, i = depth + 1, CAT_OPEN, i + 1
            continue

        if c == ")":
            if CAT_CLOSE not in allowed or depth == 0:
                return None
            depth, last, i = depth - 1, CAT_CLOSE, i + 1
            continue

        if c == ".":
            if CAT_DOT not in allowed:
                return None
            last, i = CAT_DOT, i + 1
            continue

        if c == "[":
            if CAT_ATOM not in allowed:
                return None
            stage, i = _B_START, i + 1
            continue

        if c in BONDS:
            if CAT_BOND not in allowed:
                return None
            last, i = CAT_BOND, i + 1
            continue

        if c == "*":
            if CAT_ATOM not in allowed:
                return None
            rings = _age_rings(rings, depth)
            last, i = CAT_ATOM, i + 1
            continue

        if c.isupper():
            if CAT_ATOM not in allowed:
                return None
            step = 2 if text[i : i + 2] in _TWO_CHAR_ORGANIC else (1 if c in ORGANIC else 0)
            if not step:
                return None
            rings = _age_rings(rings, depth)
            last, i = CAT_ATOM, i + step
            continue

        if c.islower():
            if CAT_ATOM not in allowed:
                return None
            step = 2 if text[i : i + 2] in _TWO_CHAR_AROMATIC else (1 if c in AROMATIC else 0)
            if not step:
                return None
            rings = _age_rings(rings, depth)
            last, i = CAT_ATOM, i + step
            continue

        return None

    return (depth, last, rings, stage, pct, chir)


class SmilesStateMachine:
    """Stateful wrapper for validating complete strings; failures are sticky."""

    def __init__(self):
        self.state = INITIAL

    def feed(self, text):
        self.state = advance(self.state, text)
        return self.state

    @property
    def valid(self):
        return self.state is not None

    @property
    def terminal(self):
        return self.state is not None and is_terminal(self.state)


class SmilesGrammarConstraint:
    """transformers LogitsProcessor enforcing SMILES syntax during generation.

    EOS is unmasked only once the partial string is terminal, which is what turns
    "usually valid" into "cannot emit unbalanced SMILES".
    """

    _RING_CHARS = frozenset("0123456789%")
    # Raw vocab pieces carry a space marker that is not a SMILES character. Decoded text
    # turns it back into a space, which advance() already skips, so normalize to match.
    _SPACE_MARKERS = ("\u2581", "\u0120")

    def __init__(self, tokenizer, eos_token_id=None):
        self.tokenizer = tokenizer
        vocab = tokenizer.get_vocab()
        self.vocab_size = max(vocab.values()) + 1
        self.token_strs = [""] * self.vocab_size
        for tok, idx in vocab.items():
            for marker in self._SPACE_MARKERS:
                tok = tok.replace(marker, " ")
            self.token_strs[idx] = tok

        self.eos_id = tokenizer.eos_token_id if eos_token_id is None else eos_token_id
        special = {
            i
            for i in (
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
                tokenizer.unk_token_id,
                getattr(tokenizer, "bos_token_id", None),
            )
            if i is not None
        }
        self._special = frozenset(special)

        self._ring_ids = tuple(
            idx
            for idx in range(self.vocab_size)
            if idx not in self._special
            and self.token_strs[idx]
            and self._RING_CHARS.intersection(self.token_strs[idx])
        )

        self._base_cache = {}
        self._final_cache = {}
        self._state_cache = {}
        self.base_builds = 0
        self.final_builds = 0

    def state_for_text(self, text):
        cached = self._state_cache.get(text)
        if cached is None:
            cached = advance(INITIAL, text) or INITIAL
            self._state_cache[text] = cached
        return cached

    def _build_base_mask(self, state):
        import torch

        ringless = (state[0], state[1], frozenset(), state[3], state[4], state[5])
        keep = torch.zeros(self.vocab_size, dtype=torch.bool)
        strs = self.token_strs
        for idx in range(self.vocab_size):
            if idx in self._special:
                continue
            s = strs[idx]
            if s and advance(ringless, s) is not None:
                keep[idx] = True
        self.base_builds += 1
        return keep

    def mask_for_state(self, state):
        key = (base_key(state), state[2], is_terminal(state))
        mask = self._final_cache.get(key)
        if mask is not None:
            return mask

        bk = base_key(state)
        base = self._base_cache.get(bk)
        if base is None:
            base = self._build_base_mask(state)
            self._base_cache[bk] = base

        mask = base.clone()
        if state[2]:
            strs = self.token_strs
            for idx in self._ring_ids:
                if mask[idx] and advance(state, strs[idx]) is None:
                    mask[idx] = False
        if self.eos_id is not None:
            mask[self.eos_id] = is_terminal(state)
        if not bool(mask.any()) and self.eos_id is not None:
            mask[self.eos_id] = True

        self._final_cache[key] = mask
        self.final_builds += 1
        return mask

    def __call__(self, input_ids, scores):
        import torch

        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        neg = torch.finfo(scores.dtype).min

        for i in range(scores.shape[0]):
            mask = self.mask_for_state(self.state_for_text(texts[i]))
            scores[i] = scores[i].masked_fill(~mask.to(scores.device), neg)

        return scores
