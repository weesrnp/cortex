from src.search import mask_pii


def test_mascara_cpf_e_cns():
    assert mask_pii("CPF 123.456.789-09") == "CPF 123.***.***-09"
    assert mask_pii("cpf=12345678909;") == "cpf=123.***.***-09;"
    assert mask_pii("CNS 700 5001 2345 6789") == "CNS 700 **** **** 6789"
    assert mask_pii("CNS 700500123456789") == "CNS 700 **** **** 6789"


def test_preserva_outros_codigos():
    texto = "CNES 1234567 | INE 0001234567 | SIGTAP 0301010064 | co_dim_tempo 20260923 | IBGE 3550308"
    assert mask_pii(texto) == texto


def test_vazio():
    assert mask_pii("") == ""
    assert mask_pii(None) is None
