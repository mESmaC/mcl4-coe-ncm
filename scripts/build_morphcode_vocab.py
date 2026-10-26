"""Build the deterministic `MorphCode-16K` (`vocab_size = 16,384`) vocabulary JSON (`src/mcl4_coe/morphcode_16k.json`).

Combines:
1. `0..255`: Raw UTF-8 single bytes (`<byte_00>` .. `<byte_ff>`) for 100% lossless byte fallback.
2. `256..259`: Special control IDs (`<pad>=256`, `<bos>=257`, `<eos>=258`, `<unk>=259`).
3. `260..275`: Atomic `MCL4-COE` trace & operator markers (`#U$#`, `#Q$#`, `#S$#`, `#H$#`, `#T$#`, `#A$#` + space/newline forms).
4. `276..465`: Code indentation, multi-char operators, Python keywords, and LaTeX math primitives.
5. `466..1,250`: Productive English + Greco-Latin morpheme prefixes (bare, space-prefixed, Title-cased),
   derivational/inflectional suffixes, and high-frequency 2–3 char syllabic subwords.
6. `1,251..16,383`: Canonical root stems and high-frequency words/identifiers mined from local shards,
   scientific/medical/math/code lexicons, and rank-ordered GPT-2 vocabulary (with multi-digit numbers
   and redundant regular inflections stripped so `16,384` slots cover the root vocabulary of a ~50K BPE).
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mcl4_coe.morph_tokenizer import (
    ATOMIC_SHORT_WORDS,
    ATOMIC_TRACE_TOKENS,
    MORPHEME_PREFIXES,
    MORPHEME_SUFFIXES,
    VOCAB_JSON_PATH,
)

CODE_AND_MATH_TOKENS: tuple[str, ...] = (
    # Indentation & whitespace structure
    "\n    ",
    "\n        ",
    "\n            ",
    "\n                ",
    "\n                    ",
    "\n                        ",
    "\n\n",
    "    ",
    "        ",
    "            ",
    "                ",
    "\n\t",
    "\n\t\t",
    "  ",
    "   ",
    # Multi-character operators & delimiters (leading-space convention so following words keep their leading space)
    " ==",
    " !=",
    " <=",
    " >=",
    " ->",
    " =>",
    " +=",
    " -=",
    " *=",
    " /=",
    " //",
    " **",
    " &&",
    " ||",
    " <<",
    " >>",
    " ===",
    " !==",
    " =",
    " +",
    " -",
    " *",
    " /",
    " <",
    " >",
    "==",
    "!=",
    "<=",
    ">=",
    "->",
    "=>",
    "+=",
    "-=",
    "*=",
    "/=",
    "//",
    "**",
    ":=",
    "::",
    "...",
    '"""',
    "'''",
    "```",
    "```python\n",
    "```python",
    "```\n",
    "):\n",
    " (",
    " [",
    " {",
    # Python & software engineering structural tokens
    "def",
    "class",
    "return",
    "import",
    "from",
    " def",
    " class",
    " return",
    " import",
    " from",
    "self.",
    "self",
    "__init__",
    "__name__",
    "__main__",
    "None",
    "True",
    "False",
    " None",
    " True",
    " False",
    "async",
    "await",
    "lambda",
    "yield",
    "raise",
    "assert",
    "except",
    "finally:",
    "else:",
    "elif",
    "if",
    "for",
    "while",
    "with",
    "try:",
    "pass",
    "break",
    "continue",
    "isinstance",
    "len",
    "range",
    "enumerate",
    "zip",
    "dict",
    "list",
    "set",
    "tuple",
    "super",
    "torch.",
    "nn.",
    "F.",
    "np.",
    "math.",
    # LaTeX & mathematical primitives
    "\\frac{",
    "\\sqrt{",
    "\\sum",
    "\\int",
    "\\prod",
    "\\lim",
    "\\infty",
    "\\alpha",
    "\\beta",
    "\\gamma",
    "\\delta",
    "\\epsilon",
    "\\theta",
    "\\lambda",
    "\\mu",
    "\\pi",
    "\\sigma",
    "\\tau",
    "\\phi",
    "\\omega",
    "\\Delta",
    "\\Sigma",
    "\\Omega",
    "\\cdot",
    "\\times",
    "\\leq",
    "\\geq",
    "\\neq",
    "\\approx",
    "\\equiv",
    "\\in",
    "\\notin",
    "\\subset",
    "\\subseteq",
    "\\cup",
    "\\cap",
    "\\forall",
    "\\exists",
    "\\nabla",
    "\\partial",
    "\\boxed{",
    "\\text{",
    "\\mathrm{",
    "\\mathbf{",
    "\\left(",
    "\\right)",
    "\\left[",
    "\\right]",
    "\\begin{",
    "\\end{",
    "^{",
    "_{",
)

# Canonical bound & free scientific/technical/English stems that anchor morphological families
CORE_BOUND_AND_FREE_STEMS: tuple[str, ...] = (
    "acceler", "accumul", "accur", "achiev", "acquir", "activ", "adapt", "addit",
    "adequ", "adjust", "administr", "advanc", "advantag", "aggr", "aggreg", "algorith",
    "align", "alloc", "altern", "ambigu", "amplif", "analy", "analys", "analyt",
    "anatom", "annihil", "annot", "anticip", "appli", "applic", "appreci", "approac",
    "appropri", "approxim", "arbitr", "architect", "argu", "arithmet", "arrang", "articul",
    "assimil", "associ", "assum", "assumpt", "asymmetr", "asymptot", "atmospher", "attenu",
    "attent", "attribut", "augment", "authent", "author", "automat", "autonom", "avail",
    "averag", "balanc", "baromet", "behav", "behavior", "benefi", "bibliograph", "biolog",
    "bipart", "bound", "boundar", "calibr", "calcul", "candid", "capacit", "capillar",
    "captur", "carbon", "cardiac", "cardiovascul", "cataly", "categor", "causal", "celebr",
    "cellul", "central", "centrifug", "cerebr", "certif", "challeng", "character", "chem",
    "chemic", "chromosom", "chronolog", "chronomet", "circul", "circumfer", "classif", "clin",
    "clinic", "cluster", "coalesc", "coher", "collabor", "collaps", "collat", "collect",
    "collis", "combin", "communic", "compar", "compat", "compens", "compet", "compil",
    "complement", "complet", "complex", "complic", "compon", "compos", "composit", "comprehend",
    "comprehens", "compress", "comput", "concaten", "concentr", "concept", "conclud", "conclus",
    "concur", "condens", "condit", "conduct", "confid", "configur", "confirm", "conflict",
    "conform", "congru", "connect", "consecut", "consequ", "conserv", "consider", "consist",
    "consolid", "constant", "constitut", "constrain", "construct", "consum", "contamin", "contemp",
    "context", "continu", "contract", "contradict", "contrast", "contribut", "control", "controvers",
    "convect", "conveni", "convent", "converg", "convers", "convert", "coordin", "correl",
    "correspond", "corrobor", "cortic", "cosmolog", "counter", "covar", "covari", "creat",
    "criteri", "critic", "crystall", "cultiv", "cumul", "curat", "curricul", "curvat",
    "cybernet", "cylind", "decompos", "dedu", "deduct", "defici", "defin", "definit",
    "deform", "degener", "degrad", "deliber", "deliver", "demonstr", "denomin", "densit",
    "depend", "deplet", "deploy", "depolar", "deposit", "depreci", "depress", "deriv",
    "describ", "descript", "design", "destabil", "destroy", "destruct", "detail", "detect",
    "deterior", "determin", "develop", "deviat", "diagnos", "diagnost", "diagram", "diamet",
    "differ", "differenti", "diffract", "diffus", "digit", "dilat", "dilut", "dimens",
    "dimension", "diminish", "direct", "disabl", "disappear", "discharg", "disciplin", "disclos",
    "disconnect", "discontinu", "Jiscover", "discover", "discrimin", "discuss", "diseas", "disintegr",
    "displac", "display", "dispos", "disrupt", "Jissip", "dissip", "dissoci", "dissolv",
    "distan", "distill", "distinct", "distinguish", "distort", "distribut", "diverg", "divers",
    "divid", "divis", "document", "domin", "dopamin", "dramat", "duplic", "dynam",
    "econom", "ecosystem", "effect", "effici", "eigenval", "eigenvec", "elabor", "elast",
    "electr", "electro", "electromagnet", "electron", "element", "elev", "elimin", "ellipt",
    "elong", "elucid", "embed", "embedd", "emerg", "emiss", "emphas", "empir",
    "employ", "encapsul", "encod", "encount", "encourag", "encrypt", "endocrin", "endothel",
    "energ", "energetic", "enforc", "engag", "engine", "enhanc", "enorm", "ensembl",
    "entail", "entangl", "enterpris", "enthalp", "entrop", "enumer", "environ", "environment",
    "enzym", "epidemi", "epiderm", "epigenet", "episod", "epithel", "equal", "equat",
    "equilibr", "equip", "equival", "errone", "establ", "estim", "etern", "ethic",
    "etiolog", "euclid", "evalu", "evapor", "eventu", "evid", "evolu", "evolut",
    "exacerb", "exagger", "examin", "exampl", "excav", "exceed", "excel", "except",
    "exchang", "excit", "exclud", "exclus", "execut", "exempl", "exercis", "exhaust",
    "exhibit", "exist", "expand", "expans", "expect", "expedit", "experi", "experim",
    "experiment", "expert", "explain", "explan", "explicit", "exploit", "explor", "explos",
    "exponent", "exponenti", "export", "expos", "express", "extend", "extens", "exterior",
    "extern", "extinct", "extract", "extrapol", "extrem", "fabric", "facilit", "factor",
    "cult", "famili", "fascin", "featur", "feder", "feedback", "ferment", "ferromagnet",
    "fibrill", "figur", "filtr", "filtrat", "financ", "financi", "finit", "flexibl",
    "fluctu", "fluoresc", "focus", "formul", "formulat", "foundat", "fraction", "fragment",
    "framework", "frequen", "friction", "frontier", "function", "fundament", "further", "fusion",
    "galact", "galvan", "gastro", "gener", "general", "genet", "genom", "geometr",
    "geolog", "gestat", "global", "glomerul", "glucos", "govern", "gradi", "gradient",
    "grammat", "granul", "graphic", "gravitat", "guarant", "guidanc", "habitat", "hallucin",
    "harmon", "hazard", "hemoglob", "hemorrhag", "hepat", "heredit", "heterogen", "heurist",
    "hierarch", "histol", "histor", "homeostas", "homogen", "horizon", "hormon", "hospit",
    "human", "humid", "hybrid", "hydraul", "hydro", "hydrogen", "hydroly", "hydrophob",
    "hypertens", "hypertroph", "hypoth", "hypothes", "hypothet", "ident", "identif", "identit",
    "ideolog", "ignor", "illumin", "illustr", "imagin", "immedi", "immers", "immigr",
    "immobil", "immun", "impact", "impair", "imped", "imper", "implement", "implic",
    "implicit", "import", "impos", "imposs", "improv", "impuls", "inactiv", "inadequ",
    "incid", "inclin", "includ", "inclus", "incompat", "incomplet", "incorpor", "incorrect",
    "increas", "increment", "incub", "independ", "indic", "indirec", "individu", "indu",
    "induct", "industr", "inequal", "inerti", "inevit", "infan", "infect", "infer",
    "inferior", "infiltr", "infinit", "inflam", "inflamm", "inflat", "influenc", "inform",
    "infrastruct", "ingredi", "inhabit", "inher", "inhibit", "initi", "inject", "injur",
    "innov", "inorgan", "inquiri", "insert", "insid", "insight", "inspect", "inspir",
    "instal", "instanc", "instant", "instinct", "institut", "instruct", "instrument", "insuffici",
    "insul", "insulin", "integr", "intellect", "intellig", "intend", "intens", "intent",
    "interact", "intercept", "interchang", "interconnect", "interest", "interfac", "interfer", "interior",
    "intermedi", "intern", "interpol", "interpret", "interrog", "interrupt", "intersect", "interv",
    "interven", "interview", "intestin", "intracellul", "intraven", "intrins", "introduc", "introduct",
    "intuit", "invari", "invad", "inval", "invalid", "invas", "invent", "invers",
    "invert", "investig", "invest", "invis", "involv", "ioniz", "irradi", "irregular",
    "irrevers", "irrig", "irrit", "ischem", "isol", "isomorph", "isotop", "iter",
    "journ", "judg", "junction", "justif", "kinet", "knowledg", "laborator", "lamin",
    "languag", "laryng", "latenc", "later", "latitud", "launch", "layer", "leader",
    "learn", "legisl", "legitim", "length", "lesion", "leukocyt", "leverag", "lexic",
    "liabil", "liber", "librar", "licens", "ligament", "light", "likeli", "limit",
    "linear", "linguist", "liquid", "liter", "literat", "lithium", "locat", "locomot",
    "logarithm", "logic", "logist", "longitud", "luminesc", "lymph", "lymphocyt", "machin",
    "macro", "macrophag", "magnet", "magnif", "magnitud", "maintain", "mainten", "major",
    "malign", "mammal", "manag", "mandat", "manifest", "manifold", "manipul", "manufactur",
    "manuscript", "margin", "market", "markov", "master", "match", "materi", "mathemat",
    "matric", "matrix", "matur", "maxim", "maximum", "meaning", "measur", "mechan",
    "mediat", "medic", "medicat", "medium", "melan", "membran", "memor", "mental",
    "mention", "merchand", "metab", "metabol", "metacognit", "metall", "metamorph", "metaphor",
    "metastas", "meteor", "method", "methodolog", "metric", "micro", "microb", "microscop",
    "migrat", "militar", "mineral", "minim", "minimum", "minister", "minor", "mitochondri",
    "mitos", "mitot", "mixtur", "mobil", "modal", "model", "moder", "modern",
    "modif", "modul", "molecul", "moment", "momentum", "monitor", "monoton", "morph",
    "morpholog", "mortal", "motiv", "motor", "mount", "movem", "multipl", "multiplic",
    "multitud", "muscul", "mutat", "mutual", "myocard", "narrat", "nation", "natur",
    "navig", "necess", "necessar", "necrot", "negat", "neglect", "negoti", "neighbor",
    "neoplas", "nephron", "nervous", "network", "neural", "neuro", "neurolog", "neuron",
    "neurotransmitt", "neutral", "neutron", "nitrogen", "nomenclatur", "nomin", "nonlin", "normal",
    "notat", "notic", "notif", "notion", "novelt", "nuclear", "nuclei", "nucleotid",
    "numer", "nutri", "nutrit", "object", "oblig", "observ", "obstacl", "obstruct",
    "obtain", "occasion", "occup", "occur", "occurr", "ocean", "offens", "offici",
    "olfact", "oligomer", "oncolog", "oper", "opinion", "opportun", "oppos", "opposit",
    "optic", "optim", "option", "orbit", "orchestr", "ordin", "ordinar", "organ",
    "organis", "organiz", "orient", "origin", "orthogon", "oscill", "osmos", "osmot",
    "osteo", "outcom", "outlin", "output", "overcom", "overflow", "overlap", "overload",
    "oxid", "oxygen", "packag", "palli", "pancre", "pancreat", "paradigm", "paradox",
    "parallel", "paramet", "parameter", "parasit", "parent", "pariet", "particip", "particl",
    "particular", "partit", "partner", "passag", "passiv", "pathogen", "patholog", "patient",
    "pattern", "pediatr", "penalt", "penetr", "perceiv", "percept", "percent", "percentag",
    "perform", "peripher", "perman", "permeab", "permiss", "permit", "permut", "perpendicul",
    "perpetu", "persist", "person", "perspect", "perturb", "pervas", "pharmaceut", "pharmacolog",
    "pharyng", "phenomen", "phenotyp", "philosoph", "phosphat", "phosphor", "phosphoryl", "photon",
    "photosynthes", "phras", "physic", "physiolog", "pictur", "pipelin", "pituitar", "placeb",
    "placent", "planet", "plant", "plasma", "plastic", "platelet", "platform", "plausibl",
    "pleur", "pneumon", "point", "poison", "polar", "polic", "polit", "pollut",
    "polymer", "polynomi", "popul", "popular", "portion", "posit", "possess", "possibl",
    "posterior", "postul", "potass", "potenti", "practic", "pragmat", "preced", "precipit",
    "precis", "preclud", "precursor", "predat", "predecess", "predict", "predomin", "prefer",
    "prefix", "pregnan", "preliminar", "prepar", "preponder", "prescrib", "prescript", "presenc",
    "present", "preserv", "presid", "pressur", "presum", "prevail", "preval", "prevent",
    "previous", "primar", "primit", "princip", "principl", "priorit", "privac", "privileg",
    "probab", "probabil", "problem", "procedur", "proceed", "process", "proclaim", "produc",
    "product", "profess", "profession", "profil", "profit", "profound", "prognos", "program",
    "programm", "progress", "prohibit", "project", "prolifer", "prolong", "promin", "promis",
    "promot", "prompt", "pronounc", "propag", "propens", "proper", "properti", "prophylax",
    "proport", "propos", "proposit", "propriet", "propuls", "prospect", "prosper", "prostat",
    "protect", "protein", "proteolyt", "protocol", "proton", "prototyp", "provid", "provinc",
    "provis", "provok", "proxim", "psychiatr", "psycholog", "public", "publish", "pulmon",
    "pulmonar", "punctu", "purchas", "purif", "purpos", "pursu", "pyram", "python",
    "quadrat", "qualif", "qualit", "quantif", "quantit", "quantiz", "quantum", "quarantin",
    "quarter", "quatern", "queri", "question", "quotient", "radiat", "radic", "radioact",
    "random", "rapid", "ration", "rational", "react", "reactiv", "reactor", "reader",
    "readi", "realist", "realit", "realiz", "reason", "reassur", "receiv", "recent",
    "recept", "receptor", "recess", "recip", "recipro", "recogn", "recognit", "recommend",
    "reconcil", "reconstruct", "record", "recover", "recreat", "recruit", "rectangul", "recur",
    "recurr", "recurs", "recycl", "redemp", "redirect", "redistribut", "reduc", "reduct",
    "redund", "refer", "referenc", "refin", "reflect", "reflex", "reform", "refract",
    "refresh", "refriger", "regard", "regener", "regim", "region", "regist", "regress",
    "regul", "regular", "rehabilit", "reinforc", "reject", "relat", "relativ", "relax",
    "releas", "relev", "reliab", "reliev", "relig", "relinquish", "remain", "remark",
    "remedi", "rememb", "remind", "remiss", "remot", "remov", "renal", "render",
    "renew", "reorgan", "repair", "repeat", "repertoir", "repetit", "replac", "replenish",
    "replic", "report", "repositor", "repres", "represent", "repress", "reproduc", "reproduct",
    "reput", "request", "requir", "resc", "research", "resembl", "reserv", "reservoir",
    "resid", "residu", "resili", "resist", "resolut", "resolv", "reson", "resourc",
    "respect", "respir", "respirat", "respond", "respons", "restor", "restrain", "restrict",
    "result", "resum", "retain", "retent", "retin", "retir", "retriev", "retrospect",
    "return", "reveal", "revenu", "revers", "review", "revis", "reviv", "revolut",
    "reward", "rhythm", "ribosom", "rigid", "rigor", "ritual", "robust", "rotat",
    "rout", "router", "routin", "ruptur", "sacrific", "safeguard", "salient", "saliv",
    "sampl", "sanit", "satellit", "satisf", "satisfact", "satur", "scaffold", "scalar",
    "scale", "scann", "scatter", "scenari", "schedul", "schema", "schemat", "scholar",
    "scienc", "scient", "scientif", "scler", "scratch", "scratchpad", "screen", "script",
    "scrutin", "search", "season", "second", "secondar", "secret", "section", "sector",
    "secur", "sediment", "segment", "segreg", "seizur", "select", "semant", "semiconduct",
    "seminar", "senesc", "sensit", "sensor", "sentenc", "separ", "sequenc", "sequenti",
    "sequest", "serial", "serious", "serum", "servant", "server", "servic", "session",
    "settl", "sever", "shadow", "shallow", "shard", "shelter", "shield", "shift",
    "shortag", "shoulder", "shrink", "shuttle", "sibling", "side", "siev", "signal",
    "signatur", "signif", "signific", "resembl", "silic", "silicon", "similar", "simpl",
    "simplif", "simul", "simultan", "sincer", "singul", "singular", "situat", "skelet",
    "skeleton", "skeptic", "skill", "slender", "smooth", "social", "societ", "sodium",
    "softwar", "solar", "soldier", "solemn", "solid", "solidif", "solitar", "solub",
    "solut", "solvent", "somat", "sophist", "sort", "sound", "sourc", "sovereign",
    "space", "spatial", "speaker", "special", "speci", "specif", "specifi", "specimen",
    "spectr", "spectroscop", "specul", "speech", "speed", "spher", "spheric", "spinal",
    "spirit", "splen", "split", "spontane", "sponsor", "spread", "spring", "square",
    "stabil", "stabl", "stagn", "standard", "start", "statem", "statist", "status",
    "statut", "steadi", "stellar", "stenos", "stereo", "steril", "steroid", "stimul",
    "stochast", "stoichiometr", "stomach", "storag", "store", "straight", "strain", "strand",
    "strateg", "strategi", "stratif", "stream", "strength", "strengthen", "stress", "stretch",
    "strict", "strike", "string", "strip", "stroke", "strong", "structur", "struggl",
    "student", "studi", "style", "subject", "sublim", "submerg", "submit", "subordin",
    "subsequ", "subsid", "subsidi", "substanc", "substanti", "substitut", "substrat", "subtl",
    "subtract", "suburb", "succeed", "success", "succinct", "suffer", "suffici", "suffix",
    "suggest", "suitabl", "sulfur", "summar", "summari", "summit", "superfici", "superior",
    "supervis", "supplem", "supplement", "suppli", "support", "suppos", "suppress", "suprem",
    "surfac", "surgeon", "surgeri", "surpass", "surplus", "surpris", "surrender", "surround",
    "survey", "surviv", "suscept", "suspect", "suspend", "suspens", "sustain", "sutur",
    "swallow", "switch", "syllab", "symbol", "symmetr", "sympath", "sympathet", "symptom",
    "synaps", "synapt", "synchron", "syndrom", "synerg", "synonym", "syntact", "syntax",
    "synthes", "synthet", "system", "systemat", "systol", "tabul", "tactic", "tailor",
    "talent", "tangent", "tangibl", "target", "taxonom", "teacher", "techn", "technic",
    "techniqu", "technolog", "tecton", "telecommun", "telescop", "televis", "temper", "temperatur",
    "templat", "tempor", "temporar", "tendenc", "tender", "tendon", "tension", "tensor",
    "tentat", "termin", "terminolog", "ternar", "terrestri", "territor", "tertiar", "testimoni",
    "tetrah", "textur", "thalam", "theat", "themat", "theolog", "theorem", "theoret",
    "theori", "therap", "therapeut", "thermal", "thermodynam", "thermomet", "thesaur", "thesis",
    "thick", "think", "thorough", "thought", "thousand", "threat", "threaten", "threshold",
    "throat", "thrombos", "through", "put", "thunder", "thyroid", "ticket", "tight",
    "tighten", "timber", "timelin", "tissu", "titrat", "token", "toler", "tomograph",
    "tonsil", "topolog", "torqu", "total", "touch", "radit", "tradit", "traffic",
    "traged", "train", "trajector", "transact", "transcrib", "transcript", "transduc", "transduct",
    "transfer", "transform", "transfus", "transi", "transient", "transit", "translat", "transluc",
    "transmiss", "transmit", "transmitt", "transpar", "transplant", "transport", "transpos", "transvers",
    "traum", "traumat", "travel", "travers", "treasur", "treat", "treatment", "tremend",
    "trend", "triangl", "triangul", "tribun", "tribut", "trigger", "trigonometr", "triumph",
    "trivial", "trophic", "tropic", "troubl", "truncat", "trust", "truth", "tuberculos",
    "tubul", "tumor", "tunnel", "turbul", "turn", "tutori", "twilight", "twist",
    "typic", "ultim", "ultrasound", "ultraviolet", "unanim", "uncertain", "uncondit", "unconsci",
    "underGO", "underli", "underlin", "undermin", "understand", "undertak", "unexpect", "unif",
    "uniform", "unify", "unilater", "union", "uniqu", "unit", "univers", "unknown",
    "unusu", "updat", "upgrad", "uphold", "upper", "upright", "upward", "uranium",
    "urban", "urgenc", "urgent", "urinar", "usag", "useful", "usual", "util",
    "utilit", "utiliz", "utter", "vacanc", "vaccin", "vacuum", "vaginal", "vagu",
    "valenc", "valid", "valley", "valu", "valuabl", "valve", "vanish", "vapor",
    "vari", "variabl", "varianc", "variant", "variat", "varieti", "various", "vascul",
    "vasodil", "vector", "veget", "vehicl", "veloc", "velvet", "vendor", "venous",
    "ventil", "ventricl", "ventricul", "ventur", "verbal", "verdict", "verif", "verifi",
    "versatil", "version", "vertebr", "vertex", "vertic", "vesicl", "vessel", "vestibul",
    "veteran", "viabl", "vibrat", "vicin", "victim", "victor", "video", "vigil",
    "vigor", "villag", "violat", "violenc", "violent", "viral", "virtual", "virtu",
    "virul", "virus", "visc", "viscos", "visibl", "vision", "visit", "visual",
    "vital", "vitamin", "vivid", "vocabular", "vocal", "volatil", "volcan", "voltag",
    "volum", "voluntar", "volunt", "vortex", "vulner", "wander", "warmth", "warn",
    "warrant", "water", "wave", "wavelength", "weaken", "wealth", "weapon", "weather",
    "weight", "welcom", "welfar", "western", "whatsoever", "wheel", "wherea", "whereby",
    "whether", "whisper", "whole", "wholesal", "widespread", "width", "wilder", "willing",
    "window", "winter", "wisdom", "withdraw", "withhold", "within", "without", "witness",
    "wonder", "wooden", "worker", "workflow", "workspac", "world", "worri", "worship",
    "worth", "worthi", "wound", "wrapper", "writer", "writing", "wrong", "xenon",
    "yield", "young", "youth", "zeal", "zero",
)


def _is_multi_digit_or_noisy(piece: str) -> bool:
    """Reject tokens that contain any digit (except standalone ' 0'..' 9' if desired; we keep digits 0..9 as 48..57)."""
    if any(ch.isdigit() for ch in piece):
        return True
    # Must be valid printable ASCII or standard punctuation
    for ch in piece:
        o = ord(ch)
        if o < 32 and ch not in ("\n", "\t"):
            return True
        if o >= 127:
            return True
    return False


def _mine_local_words() -> Counter[str]:
    """Mine word frequencies from local repo code, docs, and `data_shards`."""
    counts: Counter[str] = Counter()
    word_re = re.compile(r" ?[a-zA-Z]{2,20}")

    # 1. Scan repo Python and Markdown files
    for ext in ("*.py", "*.md"):
        for p in REPO_ROOT.rglob(ext):
            if ".git" in p.parts or "__pycache__" in p.parts:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                counts.update(word_re.findall(text))
            except Exception:
                pass

    # 2. Scan local uint16 byte shards (`data_shards/**/*.bin`)
    shards_dir = REPO_ROOT / "data_shards"
    if shards_dir.is_dir():
        for bin_path in shards_dir.rglob("*.bin"):
            try:
                raw = np.fromfile(bin_path, dtype="<u2") & 0x7FFF
                byte_vals = raw[(raw < 128) & (raw >= 9)].astype(np.uint8)
                text = byte_vals.tobytes().decode("ascii", errors="ignore")
                counts.update(word_re.findall(text))
            except Exception:
                pass

    # 3. Scan `decision_examples_cache.pt`
    for pt_path in shards_dir.rglob("*.pt"):
        try:
            examples = torch.load(pt_path, map_location="cpu", weights_only=False)
            if isinstance(examples, list):
                for ex in examples:
                    if hasattr(ex, "context"):
                        counts.update(word_re.findall(str(ex.context)))
                    if hasattr(ex, "descriptions"):
                        for d in ex.descriptions:
                            counts.update(word_re.findall(str(d)))
        except Exception:
            pass

    return counts


def _extract_gpt2_ranked_pieces() -> list[str]:
    """Extract rank-ordered English/code tokens from cached GPT-2 tokenizer if available."""
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        tok = AutoTokenizer.from_pretrained("gpt2")
        # Sort by token ID (0..50256), which in BPE is exact merge-rank order!
        vocab = tok.get_vocab()
        ordered = sorted(vocab.items(), key=lambda kv: kv[1])
        pieces: list[str] = []
        for raw_tok, _ in ordered:
            decoded = tok.convert_tokens_to_string([raw_tok])
            if decoded and not _is_multi_digit_or_noisy(decoded):
                pieces.append(decoded)
        return pieces
    except Exception:
        return []


def build_morphcode_16k_vocab() -> dict:
    pieces: list[str] = []
    seen: set[str] = set()
    stem_registry: set[str] = set()

    def add_piece(p: str) -> bool:
        if len(pieces) >= 16_384:
            return False
        if not p or p in seen:
            return False
        seen.add(p)
        pieces.append(p)
        return True

    # 1. IDs 0..255: Raw UTF-8 single byte placeholders
    for b in range(256):
        pieces.append(f"<byte_{b:02x}>")

    # 2. IDs 256..259: Special control IDs
    pieces.extend(["<pad>", "<bos>", "<eos>", "<unk>"])
    assert len(pieces) == 260

    # 3. IDs 260..275: Atomic MCL4-COE trace markers
    for m in ATOMIC_TRACE_TOKENS:
        add_piece(m)
    assert len(pieces) == 276

    # 4. Code indentation, multi-char operators, Python keywords & LaTeX primitives
    for tok in CODE_AND_MATH_TOKENS:
        add_piece(tok)

    # 5. Morpheme prefixes (space-prefixed, bare, and Title-cased) + Morpheme suffixes
    for pref in MORPHEME_PREFIXES:
        add_piece(" " + pref)
        add_piece(pref)
        add_piece(" " + pref.capitalize())
        add_piece(pref.capitalize())

    for suf in MORPHEME_SUFFIXES:
        if not _is_multi_digit_or_noisy(suf):
            add_piece(suf)

    # 6. Closed-class function words & atomic short words (space-prefixed, bare, Title-cased)
    for w in sorted(ATOMIC_SHORT_WORDS):
        stem_registry.add(w.lower())
        add_piece(" " + w)
        add_piece(w)
        add_piece(" " + w.capitalize())
        add_piece(w.capitalize())

    # 7. Core bound and free scientific/technical/English stems (space-prefixed & bare)
    extra_bound_stems = (
        "hes", "gravit", "gravity", "tube", "xylem", "phloem", "adhes", "cohes",
        "binary", "search", "solve", "narrow", "upward", "downward",
    )
    for st in (*CORE_BOUND_AND_FREE_STEMS, *extra_bound_stems):
        clean_st = st.lower().strip()
        if len(clean_st) >= 3 and clean_st.isalpha():
            stem_registry.add(clean_st)
            add_piece(" " + clean_st)
            add_piece(clean_st)

    # 8. High-frequency 2- and 3-letter syllabic subword building blocks (so rare words never drop to 1-char bytes)
    vowels = "aeiouy"
    consonants = "bcdfghjklmnpqrstvwxz"
    common_digraphs = (
        "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es", "or", "te",
        "of", "ed", "is", "it", "al", "ar", "st", "to", "nt", "ng", "se", "ha", "as", "ou",
        "io", "le", "ve", "co", "me", "de", "hi", "ri", "ro", "ic", "ne", "ea", "ra", "ce",
        "li", "ch", "ll", "be", "ma", "si", "om", "ur", "ca", "el", "ta", "la", "ns", "di",
        "fo", "ho", "pe", "ec", "pr", "no", "ct", "us", "ac", "ot", "il", "tr", "ly", "nc",
        "et", "ut", "ss", "so", "rs", "un", "lo", "wa", "ge", "ie", "wh", "ee", "wi", "em",
        "ad", "ol", "rt", "po", "we", "na", "ul", "ni", "ts", "mo", "ow", "pa", "im", "mi",
        "ai", "sh", "ir", "su", "id", "os", "iv", "ia", "am", "fi", "ci", "vi", "pl", "ig",
        "tu", "ev", "ld", "ry", "mp", "fe", "bl", "ab", "gh", "oo", "sp", "sc", "cl", "cr",
        "dr", "fl", "fr", "gl", "gr", "ph", "kn", "qu", "sk", "sl", "sm", "sn", "sw", "tw",
        "wr", "str", "spr", "scr", "spl", "chr", "phr", "thr", "sch",
    )
    for dg in common_digraphs:
        add_piece(dg)
        add_piece(" " + dg)

    # 9. Mine words from local codebase & shards and decompose into canonical stems
    local_counts = _mine_local_words()
    gpt2_pieces = _extract_gpt2_ranked_pieces()

    # Helper to strip regular inflectional suffixes and register the underlying stem
    regular_strip_suffixes = (
        "ations", "itions", "ation", "ition", "ating", "iting", "izing", "ising",
        "ingly", "ments", "ances", "ences", "ities", "ously", "ively", "fully",
        "ment", "ness", "able", "ible", "ated", "ates", "ized", "izes", "ical",
        "ally", "ious", "eous", "less", "ings", "ing", "est", "ism", "ist",
        "ies", "ied", "ers", "ors", "ed", "er", "or", "ly", "es", "s",
    )

    def try_extract_stem(bare_w: str) -> str | None:
        lw = bare_w.lower()
        if not lw.isalpha() or len(lw) < 4:
            return None
        for suf in regular_strip_suffixes:
            if len(lw) >= len(suf) + 4 and lw.endswith(suf):
                cand = lw[: -len(suf)]
                if len(cand) >= 4:
                    return cand
        return lw

    # Add stems and high-frequency words from our local corpus + repo
    for word_tok, _ in local_counts.most_common(4500):
        if _is_multi_digit_or_noisy(word_tok):
            continue
        bare = word_tok.lstrip(" ")
        if not bare.isalpha() or len(bare) < 2:
            continue
        stem = try_extract_stem(bare)
        if stem and len(stem) >= 3:
            stem_registry.add(stem)
            add_piece(" " + stem)
        add_piece(word_tok)

    # 10. Fill remaining slots up to 16,384 from rank-ordered GPT-2 vocabulary,
    #     prioritizing canonical root stems and skipping redundant regular inflections
    #     when their `stem + suffix` decomposition is already in the vocabulary!
    suffix_check_set = set(MORPHEME_SUFFIXES)
    for rank_idx, cand in enumerate(gpt2_pieces):
        if len(pieces) >= 16_384:
            break
        if _is_multi_digit_or_noisy(cand):
            continue
        has_sp = cand.startswith(" ")
        bare = cand[1:] if has_sp else cand
        if not bare:
            continue

        # If it's an alphabetic word, check if it's a redundant regular inflection of an already-known stem
        if bare.isalpha() and len(bare) >= 4:
            lw = bare.lower()
            decomposed = False
            for suf in regular_strip_suffixes:
                if len(lw) >= len(suf) + 3 and lw.endswith(suf) and suf in suffix_check_set:
                    st = lw[: -len(suf)]
                    st_cased = bare[: -len(suf)]
                    st_tok = (" " + st_cased) if has_sp else st_cased
                    # Ensure the root stem itself is in the vocabulary!
                    stem_registry.add(st)
                    add_piece(st_tok)
                    if rank_idx < 3500:
                        add_piece(st)
                    if st_tok in seen and suf in seen:
                        decomposed = True
                        break
            if decomposed:
                # Skip the inflected surface form so we save the vocab slot for a new root word!
                continue
            if bare.islower() and len(bare) >= 3:
                stem_registry.add(lw)
                add_piece(cand)
                if rank_idx < 2500:
                    add_piece(bare)
                continue

        add_piece(cand)

    # Second pass without skipping inflections if any slots remain < 16,384
    for cand in gpt2_pieces:
        if len(pieces) >= 16_384:
            break
        if not _is_multi_digit_or_noisy(cand):
            add_piece(cand)

    # Deterministic fallback fill if GPT-2 cache wasn't enough (should never trigger since GPT-2 has 50K)
    idx_fill = 0
    while len(pieces) < 16_384:
        c1 = consonants[idx_fill % len(consonants)]
        v1 = vowels[(idx_fill // len(consonants)) % len(vowels)]
        c2 = consonants[(idx_fill // (len(consonants) * len(vowels))) % len(consonants)]
        v2 = vowels[(idx_fill // (len(consonants) ** 2 * len(vowels))) % len(vowels)]
        add_piece(f" {c1}{v1}{c2}{v2}")
        add_piece(f"{c1}{v1}{c2}{v2}")
        idx_fill += 1

    assert len(pieces) == 16_384, f"Expected 16384 pieces, got {len(pieces)}"
    return {
        "format": "morphcode_16k_v1",
        "vocab_size": 16_384,
        "pad_id": 256,
        "bos_id": 257,
        "eos_id": 258,
        "unk_id": 259,
        "pieces": pieces,
        "stems": sorted(stem_registry),
    }


def main() -> None:
    payload = build_morphcode_16k_vocab()
    VOCAB_JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=None, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"Saved MorphCode-16K vocabulary to {VOCAB_JSON_PATH} "
        f"(vocab_size={len(payload['pieces'])}, stems={len(payload['stems'])}, "
        f"size={VOCAB_JSON_PATH.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
