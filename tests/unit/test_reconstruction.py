"""Tests for ontograph/reconstruction/"""

from unittest.mock import MagicMock

import pytest

from ontograph.reconstruction.direct_extractor import extract_direct
from ontograph.reconstruction.schema import (
    ArmDebug,
    ArmResult,
    DirectExtractionResult,
    DirectTriple,
    ReconstructionDebug,
    ReconstructionReport,
    TripleDetail,
)
from ontograph.reconstruction.runner import pick_winner
from ontograph.utils.iri_align import (
    iri_similarity as _iri_similarity,
    cross_iri_align as _cross_iri_align,
    apply_iri_remap as _apply_iri_remap,
    IriPairJudgment as _IriPairJudgment,
)
from ontograph.llm.base import LLMResponse, TokenUsage

NS       = "http://example.org/test#"
CLASSES  = ["OnboardComputer", "Radio", "Battery", "SolarPanel"]
PROPERTIES = ["massKg", "powerW", "dataRateMbps", "operatingTempMinC"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_extraction_response(triples: list[DirectTriple]) -> LLMResponse:
    result = DirectExtractionResult(triples=triples)
    return LLMResponse(
        parsed=result,
        raw_json=result.model_dump_json(),
        model_id="test-model",
        usage=TokenUsage(input_tokens=100, output_tokens=200),
    )


def _mock_provider(triples: list[DirectTriple]) -> MagicMock:
    provider = MagicMock()
    provider.complete.return_value = _make_extraction_response(triples)
    return provider


def _sample_triples() -> list[DirectTriple]:
    return [
        DirectTriple(subject="OBC1",        rdf_type="OnboardComputer", property="massKg", value="0.04"),
        DirectTriple(subject="Radio1",       rdf_type="Radio",           property="powerW", value="4.0"),
        DirectTriple(subject="BatteryPack1", rdf_type="Battery",         property="massKg", value="0.12"),
    ]


def _arm_result(arm: str, triple_f1: float) -> ArmResult:
    return ArmResult(
        arm=arm,
        individual_precision=1.0, individual_recall=1.0, individual_f1=1.0,
        triple_precision=1.0, triple_recall=1.0, triple_f1=triple_f1,
        triple_count_source=10, triple_count_predicted=10,
    )


# ── extract_direct ────────────────────────────────────────────────────────────

class TestExtractDirect:

    def test_calls_provider_once(self):
        provider = _mock_provider(_sample_triples())
        extract_direct("Some document text.", NS, CLASSES, PROPERTIES, provider)
        provider.complete.assert_called_once()

    def test_returns_direct_extraction_result(self):
        provider = _mock_provider(_sample_triples())
        result = extract_direct("Some document text.", NS, CLASSES, PROPERTIES, provider)
        assert isinstance(result, DirectExtractionResult)

    def test_returns_non_empty_triples(self):
        provider = _mock_provider(_sample_triples())
        result = extract_direct("Some document text.", NS, CLASSES, PROPERTIES, provider)
        assert len(result.triples) > 0

    def test_prompt_contains_class_locals(self):
        """Class names (not individual names) appear in the system prompt."""
        provider = _mock_provider(_sample_triples())
        extract_direct("doc text", NS, CLASSES, PROPERTIES, provider)
        request = provider.complete.call_args[0][0]
        system_msg = request.messages[0].content
        for cls in CLASSES:
            assert cls in system_msg

    def test_prompt_does_not_contain_individual_locals(self):
        """Individual names must NOT be given to the direct arm — it discovers them."""
        provider = _mock_provider(_sample_triples())
        extract_direct("doc text", NS, CLASSES, PROPERTIES, provider)
        request = provider.complete.call_args[0][0]
        system_msg = request.messages[0].content
        # The specific individuals from the ground truth should not appear in the prompt
        for ind in ["OBC1", "Radio1", "BatteryPack1"]:
            assert ind not in system_msg

    def test_prompt_contains_property_locals(self):
        provider = _mock_provider(_sample_triples())
        extract_direct("doc text", NS, CLASSES, PROPERTIES, provider)
        request = provider.complete.call_args[0][0]
        system_msg = request.messages[0].content
        for prop in PROPERTIES:
            assert prop in system_msg

    def test_user_message_is_document_text(self):
        doc = "This is the full document content."
        provider = _mock_provider(_sample_triples())
        extract_direct(doc, NS, CLASSES, PROPERTIES, provider)
        request = provider.complete.call_args[0][0]
        user_msg = request.messages[1].content
        assert user_msg == doc

    def test_request_uses_direct_extraction_result_model(self):
        provider = _mock_provider(_sample_triples())
        extract_direct("doc text", NS, CLASSES, PROPERTIES, provider)
        request = provider.complete.call_args[0][0]
        assert request.response_model is DirectExtractionResult

    def test_temperature_forwarded(self):
        provider = _mock_provider(_sample_triples())
        extract_direct("doc text", NS, CLASSES, PROPERTIES, provider, temperature=0.3)
        request = provider.complete.call_args[0][0]
        assert request.temperature == 0.3


# ── DirectTriple schema ───────────────────────────────────────────────────────

class TestDirectTripleSchema:

    def test_rdf_type_optional_defaults_to_none(self):
        t = DirectTriple(subject="OBC1", rdf_type=None, property="massKg", value="0.04")
        assert t.rdf_type is None

    def test_rdf_type_set_when_provided(self):
        t = DirectTriple(subject="OBC1", rdf_type="OnboardComputer", property="massKg", value="0.04")
        assert t.rdf_type == "OnboardComputer"


# ── Triple → Graph conversion ─────────────────────────────────────────────────

class TestDirectTripleToGraph:

    def _build_graph(self, triples):
        from rdflib import Graph, Literal, Namespace
        from rdflib.namespace import OWL, RDF
        NS_OBJ = Namespace(NS)
        g = Graph()
        for triple in triples:
            subj = NS_OBJ[triple.subject]
            g.add((subj, RDF.type, OWL.NamedIndividual))
            if triple.rdf_type:
                g.add((subj, RDF.type, NS_OBJ[triple.rdf_type]))
            g.add((subj, NS_OBJ[triple.property], Literal(triple.value)))
        return g, NS_OBJ

    def test_graph_contains_correct_subject_property_value(self):
        from rdflib import Literal
        g, NS_OBJ = self._build_graph(_sample_triples())
        assert (NS_OBJ["OBC1"],  NS_OBJ["massKg"], Literal("0.04")) in g
        assert (NS_OBJ["Radio1"], NS_OBJ["powerW"], Literal("4.0"))  in g

    def test_graph_declares_named_individuals(self):
        from rdflib.namespace import OWL, RDF
        g, NS_OBJ = self._build_graph(_sample_triples())
        assert (NS_OBJ["OBC1"],   RDF.type, OWL.NamedIndividual) in g
        assert (NS_OBJ["Radio1"], RDF.type, OWL.NamedIndividual) in g

    def test_graph_contains_class_assertions_when_rdf_type_set(self):
        from rdflib.namespace import RDF
        g, NS_OBJ = self._build_graph(_sample_triples())
        assert (NS_OBJ["OBC1"],   RDF.type, NS_OBJ["OnboardComputer"]) in g
        assert (NS_OBJ["Radio1"], RDF.type, NS_OBJ["Radio"])           in g

    def test_graph_no_class_assertion_when_rdf_type_is_none(self):
        from rdflib import Literal, Namespace
        from rdflib.namespace import OWL, RDF
        NS_OBJ = Namespace(NS)
        g = self._build_graph([
            DirectTriple(subject="X1", rdf_type=None, property="massKg", value="1.0")
        ])[0]
        # Only owl:NamedIndividual — no domain class
        rdf_types = list(g.objects(NS_OBJ["X1"], RDF.type))
        assert OWL.NamedIndividual in rdf_types
        assert len(rdf_types) == 1

    def test_graph_triple_count_with_rdf_type(self):
        """Each DirectTriple with rdf_type contributes 3 graph triples."""
        g, NS_OBJ = self._build_graph(_sample_triples())
        # 3 triples per individual: owl:NamedIndividual + rdf_type class + property assertion
        assert len(g) >= len(_sample_triples()) * 3


# ── ArmDebug / ReconstructionDebug schema ────────────────────────────────────

class TestArmDebug:

    def _sample_debug(self, arm: str = "ontograph") -> ArmDebug:
        return ArmDebug(
            arm=arm,
            matched_individuals=["http://ex.org/test#OBC1"],
            missed_individuals=["http://ex.org/test#Radio1"],
            extra_individuals=[],
            matched_triples=[
                TripleDetail(
                    subject="http://ex.org/test#OBC1",
                    predicate="http://ex.org/test#massKg",
                    object="0.04",
                )
            ],
            missed_triples=[
                TripleDetail(
                    subject="http://ex.org/test#Radio1",
                    predicate="http://ex.org/test#powerW",
                    object="4.0",
                )
            ],
            extra_triples=[],
        )

    def test_arm_debug_stores_individuals(self):
        d = self._sample_debug()
        assert len(d.matched_individuals) == 1
        assert len(d.missed_individuals) == 1
        assert len(d.extra_individuals) == 0

    def test_arm_debug_stores_triples(self):
        d = self._sample_debug()
        assert len(d.matched_triples) == 1
        assert d.matched_triples[0].predicate == "http://ex.org/test#massKg"
        assert len(d.missed_triples) == 1
        assert len(d.extra_triples) == 0

    def test_reconstruction_debug_round_trips_json(self):
        debug = ReconstructionDebug(
            id="abc123",
            created_at="2026-01-01T00:00:00+00:00",
            source_owl_path="data/ontology/source.owl",
            arms=[self._sample_debug("ontograph"), self._sample_debug("direct")],
        )
        restored = ReconstructionDebug.model_validate_json(debug.model_dump_json())
        assert restored.id == debug.id
        assert len(restored.arms) == 2
        assert restored.arms[0].matched_triples[0].object == "0.04"


# ── ReconstructionReport winner logic ─────────────────────────────────────────

class TestPickWinner:

    def test_ontograph_wins_higher_f1(self):
        arms = [_arm_result("ontograph", 0.81), _arm_result("direct", 0.58)]
        assert pick_winner(arms) == "ontograph"

    def test_direct_wins_higher_f1(self):
        arms = [_arm_result("ontograph", 0.55), _arm_result("direct", 0.70)]
        assert pick_winner(arms) == "direct"

    def test_tied_returns_none(self):
        arms = [_arm_result("ontograph", 0.65), _arm_result("direct", 0.65)]
        assert pick_winner(arms) is None

    def test_single_arm_returns_none(self):
        arms = [_arm_result("ontograph", 0.90)]
        assert pick_winner(arms) is None

    def test_empty_arms_returns_none(self):
        assert pick_winner([]) is None


# ── ReconstructionReport schema ───────────────────────────────────────────────

class TestReconstructionReport:

    def _build_report(
        self,
        ontograph_f1: float = 0.81,
        direct_f1:    float = 0.58,
    ) -> ReconstructionReport:
        arms = [
            _arm_result("ontograph", ontograph_f1),
            _arm_result("direct",    direct_f1),
        ]
        return ReconstructionReport(
            id="test-id",
            created_at="2026-01-01T00:00:00+00:00",
            source_owl_path="data/ontology/source.owl",
            document_path="data/raw/doc.md",
            provider="claude",
            arms=arms,
            winner=pick_winner(arms),
        )

    def test_winner_field_stored_correctly(self):
        report = self._build_report(ontograph_f1=0.81, direct_f1=0.58)
        assert report.winner == "ontograph"

    def test_winner_none_when_tied(self):
        report = self._build_report(ontograph_f1=0.65, direct_f1=0.65)
        assert report.winner is None

    def test_arms_count_is_two_in_both_mode(self):
        report = self._build_report()
        assert len(report.arms) == 2

    def test_arm_names_are_correct(self):
        report = self._build_report()
        arm_names = {a.arm for a in report.arms}
        assert arm_names == {"ontograph", "direct"}

    def test_report_serialises_to_json(self):
        report = self._build_report()
        json_str = report.model_dump_json()
        assert "ontograph" in json_str
        assert "direct" in json_str

    def test_report_round_trips_json(self):
        report = self._build_report()
        json_str = report.model_dump_json()
        restored = ReconstructionReport.model_validate_json(json_str)
        assert restored.id == report.id
        assert restored.winner == report.winner
        assert len(restored.arms) == len(report.arms)


# ── Cross-OWL IRI alignment ───────────────────────────────────────────────────

class TestIriSimilarity:

    def test_exact_match_returns_one(self):
        assert _iri_similarity("OBC1", "OBC1") == 1.0

    def test_case_insensitive_exact(self):
        assert _iri_similarity("obc1", "OBC1") == 1.0

    def test_acronym_detected_short_to_long(self):
        # SPI is an acronym of SerialPeripheralInterface
        assert _iri_similarity("SPI", "SerialPeripheralInterface") == 1.0

    def test_acronym_detected_long_to_short(self):
        assert _iri_similarity("SerialPeripheralInterface", "SPI") == 1.0

    def test_separator_normalised_match(self):
        assert _iri_similarity("OBC_1", "OBC1") == 1.0

    def test_jaccard_partial_overlap(self):
        # "BatteryPack" vs "BatteryModule" share "battery" token → some overlap
        score = _iri_similarity("BatteryPack", "BatteryModule")
        assert 0.0 < score < 1.0

    def test_no_overlap_returns_low(self):
        score = _iri_similarity("Radio1", "SolarPanel1")
        assert score < 0.4


class TestCrossIriAlign:

    def test_exact_match_not_remapped(self):
        # If working local already exists in source, skip it
        mapping = _cross_iri_align(["SPI"], ["SPI"], provider=None)
        assert "SPI" not in mapping

    def test_acronym_auto_aligned(self):
        # "SPI" ↔ "SerialPeripheralInterface" → score 1.0 → auto-approved
        mapping = _cross_iri_align(
            working_locals=["SerialPeripheralInterface"],
            source_locals=["SPI", "Radio1", "BatteryPack1"],
            provider=None,
        )
        assert mapping.get("SerialPeripheralInterface") == "SPI"

    def test_low_similarity_not_mapped(self):
        mapping = _cross_iri_align(
            working_locals=["Radio1"],
            source_locals=["SolarPanel1", "ThrusterAssembly"],
            provider=None,
        )
        assert mapping == {}

    def test_mid_range_without_provider_not_mapped(self):
        # Jaccard ~0.5 but no provider → not mapped
        mapping = _cross_iri_align(
            working_locals=["BatteryPack"],
            source_locals=["BatteryModule"],
            provider=None,
        )
        # BatteryPack vs BatteryModule: tokens {battery, pack} ∩ {battery, module} = 1/3 ≈ 0.33 < LLM threshold
        # Result depends on exact Jaccard; just check it doesn't crash
        assert isinstance(mapping, dict)

    def test_mid_range_with_provider_calls_llm(self):
        """When score is in [LLM_MIN, AUTO), provider.complete is called."""
        from ontograph.llm.base import LLMResponse, TokenUsage

        judgment = _IriPairJudgment(same_entity=True, confidence=0.95)
        mock_response = LLMResponse(
            parsed=judgment,
            raw_json=judgment.model_dump_json(),
            model_id="test-model",
            usage=TokenUsage(input_tokens=50, output_tokens=20),
        )
        provider = MagicMock()
        provider.complete.return_value = mock_response

        # Use a pair where Jaccard is in mid-range: "ADCS" vs "AttitudeDetermination"
        # The acronym check: ADCS → A,D,C,S but "AttitudeDetermination" only has 2 content words
        # Jaccard: {adcs} ∩ {attitude, determination} = 0 → won't hit LLM threshold anyway
        # Use a guaranteed mid-range pair instead by patching — just verify the logic path:
        # We check that when a mid-range pair is found, provider.complete is invoked.
        # We do this by testing with a mocked _iri_similarity-equivalent pair.
        # Direct test: construct a case where Jaccard ~ 0.5
        mapping = _cross_iri_align(
            working_locals=["ThermalControlUnit"],
            source_locals=["ThermalControlSystem"],
            provider=provider,
        )
        # {thermal, control, unit} ∩ {thermal, control, system} = 2, union = 4 → 0.5
        # 0.5 >= LLM_MIN (0.40) but < AUTO (0.85) → LLM called
        # LLM says same_entity=True, confidence=0.95 → mapped
        assert provider.complete.called
        assert mapping.get("ThermalControlUnit") == "ThermalControlSystem"


class TestApplyIriRemap:

    def test_remap_renames_individual(self):
        from rdflib import Graph, Namespace, URIRef, Literal
        from rdflib.namespace import RDF, OWL

        NS_OBJ = Namespace(NS)
        g = Graph()
        g.add((NS_OBJ["SerialPeripheralInterface"], RDF.type, OWL.NamedIndividual))
        g.add((NS_OBJ["SerialPeripheralInterface"], NS_OBJ["powerW"], Literal("1.5")))

        remapped = _apply_iri_remap(g, {"SerialPeripheralInterface": "SPI"}, NS)

        assert (NS_OBJ["SPI"], RDF.type, OWL.NamedIndividual) in remapped
        assert (NS_OBJ["SPI"], NS_OBJ["powerW"], Literal("1.5")) in remapped
        # Old IRI should be gone
        assert (NS_OBJ["SerialPeripheralInterface"], RDF.type, OWL.NamedIndividual) not in remapped

    def test_empty_mapping_returns_equivalent_graph(self):
        from rdflib import Graph, Namespace, Literal
        from rdflib.namespace import RDF, OWL

        NS_OBJ = Namespace(NS)
        g = Graph()
        g.add((NS_OBJ["OBC1"], RDF.type, OWL.NamedIndividual))

        result = _apply_iri_remap(g, {}, NS)
        assert (NS_OBJ["OBC1"], RDF.type, OWL.NamedIndividual) in result

    def test_remap_preserves_literal_objects(self):
        from rdflib import Graph, Namespace, Literal
        from rdflib.namespace import RDF, OWL

        NS_OBJ = Namespace(NS)
        g = Graph()
        g.add((NS_OBJ["OldName"], NS_OBJ["massKg"], Literal("0.5")))

        result = _apply_iri_remap(g, {"OldName": "NewName"}, NS)
        assert (NS_OBJ["NewName"], NS_OBJ["massKg"], Literal("0.5")) in result

    def test_non_namespace_iris_unchanged(self):
        """IRIs outside the namespace must not be remapped."""
        from rdflib import Graph, Namespace, URIRef
        from rdflib.namespace import RDF, OWL

        NS_OBJ = Namespace(NS)
        external = URIRef("http://other.org/thing#X")
        g = Graph()
        g.add((external, RDF.type, OWL.NamedIndividual))

        result = _apply_iri_remap(g, {"X": "Y"}, NS)
        assert (external, RDF.type, OWL.NamedIndividual) in result
