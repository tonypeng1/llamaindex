from types import SimpleNamespace
import re

# Local copies of the helper functions for isolated testing (avoids importing heavy deps)
NUM_MAP = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12'
}

def normalize_for_matching(q: str) -> str:
    q = q.lower()
    q = re.sub(r"\bsec\.?\b", "section", q)
    q = re.sub(r"\bfig\.?\b", "figure", q)
    q = re.sub(r"\beq\.?\b", "equation", q)
    for word, digit in NUM_MAP.items():
        q = re.sub(rf"\b{word}\b", digit, q)
    q = re.sub(r"[\.,;:\(\)\[\]']", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _find_pages_for_reference(kind: str, num: str, vector_docstore) -> list:
    pages = set()
    num = str(num)
    for _, node in getattr(vector_docstore, 'docs', {}).items():
        txt = getattr(node, 'text', '') or ''
        txt_l = txt.lower()
        if kind == 'figure':
            if re.search(rf'figure\s*[:\s]*{num}\b', txt_l) or re.search(rf'fig\.?\s*{num}\b', txt_l):
                pages.add(node.metadata.get('page'))
        elif kind == 'equation':
            if re.search(rf'equation\s*[:\s]*{num}\b', txt_l) or re.search(rf'eq\.?\s*{num}\b', txt_l):
                pages.add(node.metadata.get('page'))
            else:
                if re.search(rf'\(\s*{num}\s*\)', txt_l):
                    if 'equation' in txt_l or txt_l.strip().startswith(f'({num})'):
                        pages.add(node.metadata.get('page'))
    return sorted(pages)


def make_node(text, page):
    return SimpleNamespace(text=text, metadata={'page': page})


def test_normalize_for_matching():
    assert 'figure 2' in normalize_for_matching('Fig. 2')
    assert 'equation 3' in normalize_for_matching('Eq 3')
    assert 'section 4' in normalize_for_matching('Section four')


def test_find_pages_for_figure():
    docs = {
        'a': make_node('Figure 2: Flowchart of process', 5),
        'b': make_node('Some unrelated text', 6),
        'c': make_node('Figure 3: Another', 7),
    }
    vec = SimpleNamespace(docs=docs)
    pages = _find_pages_for_reference('figure', '2', vec)
    assert pages == [5]


def test_find_pages_for_equation_direct():
    docs = {
        'a': make_node('Equation (3): E = mc^2', 8),
        'b': make_node('Some other equation (4) inline', 9),
    }
    vec = SimpleNamespace(docs=docs)
    pages = _find_pages_for_reference('equation', '3', vec)
    assert pages == [8]


def test_find_pages_for_equation_parenthesis_with_context():
    docs = {
        'a': make_node('(5) This is equation 5 derived from ...', 10),
        'b': make_node('Text containing (5) but no eq markers', 11),
    }
    vec = SimpleNamespace(docs=docs)
    pages = _find_pages_for_reference('equation', '5', vec)
    assert pages == [10]
