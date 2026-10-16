# LOD dumps and the store

Read from the dumps of 2026-07-27 (CC0, Zenter fir d'Lëtzebuerger Sprooch).
`build.py` uses the names marked **used**; the others are in the files and are not read.

## `new_lod-tab.xml` (Flexiounstabellen, 16 MB)

```
tables
  inflection                      one table each (9,851)
    verbConjugation               @id (HUELEN1, = entry id) @model @separableVerb yes|no   (6,204)
      infinitive                  used: lemma
      pastParticiple              used: one string, variants joined "gaangen / gaang"
      auxiliaryVerb               used: hunn | sinn
      indicative
        present | pastSimple | presentPerfect | pastPerfect
          p1 .. p6
      conditional
        presentSimple | presentPerfect | pastPerfect
          p1 .. p6
      imperative
        present
          p2, p5
    adjDeclension                 @id (AL1) @number yes    (3,647)
      basicForms
        positive | comparative | superlative       "al", "méi al", "am eelsten"
      declension
        positive | comparative | superlative
          singular
            masculine | feminine | neutral
              nominative | accusative | dative | genitive
          plural
            masculineFeminineNeutral (rarely masculine | feminine | neutral)
              nominative | accusative | dative | genitive
```

- `p1`..`p6` = ech, du, hien/si/et, mir, dir, si. Imperative `p2` = singular (du), `p5` = plural (dir).
- 137 tables repeat the group `auxiliaryVerb, indicative, conditional, imperative`: once per
  auxiliary (abriechen: hunn, then sinn) or per use (plain, then reflexive). A person a use
  lacks has no element (reenen has `p3` only).
- A cell is text as LOD prints it: variants `wëllt / wëll`, an optional reflexive pronoun
  `maachen (mech)`, a separable particle `kënns un`, `!` after imperatives `géi!`; some cells
  end in a space. The live API returns the same strings.

## `new_lod-art.xml` (linguistic data, 100 MB)

```
lod
  entry                           @id (GOEN1)    (33,941)
    lemma                         used; @nRuleForm used ("benotze" for benotzen); @trademark
    ipa
    microStructure                one; two for 138 verbs (one per auxiliary or use)
      partOfSpeech                used: SUBST VRB ADJ NP ADV NB PRON PRONADV INTERJ PREP CONJ
                                  VRBPART PART ART; @gen used: M F N MN MF FN; @type (AUX, MOD,
                                  CARD, ORD, POSS, ..) not read
      inflection @type="plural"   also found inside a meaning
        form                      used; @nRuleForm used ("Ae" for Aen); @rare="true" used
      attribute                   used: register label of the whole entry
      auxiliaryVerb, pastParticiple, declensionInfo, internalLink, scientificName
      grammaticalUnit
        grammaticalInformation
        meaning                   @id @video
          number, secondaryHeadword, declensionInfo, domain, inflection, synonyms
          attribute               used: register label of this meaning
          targetLanguage          @lang de | fr | en | pt | nl
            translation           used for lang="en"
            semanticClarifier, function, definition
          examples                @count
            example               @id
              text                word, inflectedHeadword, attribute (label of the example)
              gloss
    infobox, allSynonyms, categories, WGS84, GeoJSON
```

Register labels (`attribute`): EGS colloquial, VEREELZT archaic, PEJ, FAM, VULG, GEHUEW, IRON,
KANNERSPROOCH, NEOL.

## `lod.sqlite`

| table | columns |
|---|---|
| `verbs` | `lemma, entry_id, aux, participle, p1, p2, p3, p4, p5, p6, imp_sg, imp_pl, tables_json` |
| `adjectives` | `lemma, entry_id, forms_json` |
| `headwords` | `lemma, entry_id, pos, gender, plural, n_rule_form, en_gloss, register` |
| `wordforms` | `form, lemma, kind` |

One row per `verbConjugation`, `adjDeclension` and `entry`; a lemma can have several rows
(sinn: SINN1, SINN2). Missing values are `''`.

**verbs.** Cells are stored as printed, only trimmed. `aux` joins the auxiliaries of all
groups (`hunn / sinn`). `p1`..`p6` are the present indicative, `imp_sg`/`imp_pl` the
imperative `p2`/`p5`, all taken from the group with the most present-tense persons (the first
one on a tie). `tables_json` is
`{"model", "separable", "conjugations": [{"auxiliaryVerb", "indicative": {"present": {"p1": ..}, ..}, "conditional", "imperative"}, ..]}`
with every group and tense of the dump.

**adjectives.** `lemma` is `basicForms/positive`; `forms_json` is the whole table as nested
objects named like the XML.

**headwords.** `pos` and `gender` come from the first `partOfSpeech`. `plural` joins the
distinct plural forms with ` / `, without the ones marked rare. `n_rule_form` is the lemma's
`nRuleForm`. `en_gloss` is the first English translation. `register` joins the labels of the
microStructure and, when every meaning carries a label, those of the meanings.

**wordforms.** `form` is the lookup key: the form without `(mech)` and `!`, single-spaced and
casefolded (`form_key` in `__init__.py`). `lemma` keeps LOD's casing. `kind`:

| kind | from |
|---|---|
| `lemma` | every article lemma and every verb infinitive |
| `plural` | every plural form, rare ones too |
| `n_rule` | `nRuleForm` of the lemma and of each plural form |
| `verb_form` | infinitive, participle and the cells of present, pastSimple, conditional presentSimple and imperative; a cell with variants gives one row per variant, stored whole (`kënns un`, `dinn of`, `doen of`) |
| `adj_form` | every cell of an adjective table (`alen`, `méi al`, `am eelsten`) |

Compound tenses (`hu geholl`, `géif goen`) are auxiliary + participle or infinitive; they are
in `tables_json` and not in `wordforms`.
