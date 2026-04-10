"""Tests for ontograph/generator/taxonomy.py"""

from ontograph.generator.taxonomy import AEROSPACE_TAXONOMY, PREDEFINED_DOMAINS


class TestTaxonomyIntegrity:

    def test_all_predefined_domains_have_system_classes(self):
        for domain in PREDEFINED_DOMAINS:
            assert domain in AEROSPACE_TAXONOMY.domain_system_classes
            assert len(AEROSPACE_TAXONOMY.domain_system_classes[domain]) >= 1

    def test_all_predefined_domains_have_required_subsystems(self):
        for domain in PREDEFINED_DOMAINS:
            assert domain in AEROSPACE_TAXONOMY.domain_required_subsystems
            assert len(AEROSPACE_TAXONOMY.domain_required_subsystems[domain]) >= 1

    def test_class_parents_exist(self):
        class_locals = {c.local for c in AEROSPACE_TAXONOMY.classes}
        for cls in AEROSPACE_TAXONOMY.classes:
            if cls.parent is not None:
                assert cls.parent in class_locals, (
                    f"Class '{cls.local}' has unknown parent '{cls.parent}'"
                )

    def test_no_duplicate_class_locals(self):
        locals_list = [c.local for c in AEROSPACE_TAXONOMY.classes]
        assert len(locals_list) == len(set(locals_list))

    def test_data_properties_have_valid_xsd_type(self):
        valid_types = {"decimal", "integer", "string", "boolean"}
        for dp in AEROSPACE_TAXONOMY.data_properties:
            assert dp.xsd_type in valid_types, (
                f"DataProperty '{dp.local}' has unknown xsd_type '{dp.xsd_type}'"
            )

    def test_object_property_classes_exist(self):
        class_locals = {c.local for c in AEROSPACE_TAXONOMY.classes}
        for op in AEROSPACE_TAXONOMY.object_properties:
            assert op.domain_class in class_locals, (
                f"ObjectProperty '{op.local}' has unknown domain_class '{op.domain_class}'"
            )
            assert op.range_class in class_locals, (
                f"ObjectProperty '{op.local}' has unknown range_class '{op.range_class}'"
            )

    def test_domain_system_classes_are_in_taxonomy(self):
        class_locals = {c.local for c in AEROSPACE_TAXONOMY.classes}
        for domain, classes in AEROSPACE_TAXONOMY.domain_system_classes.items():
            for cls in classes:
                assert cls in class_locals, (
                    f"Domain '{domain}' system class '{cls}' not found in taxonomy"
                )

    def test_domain_required_subsystems_are_in_taxonomy(self):
        class_locals = {c.local for c in AEROSPACE_TAXONOMY.classes}
        for domain, subs in AEROSPACE_TAXONOMY.domain_required_subsystems.items():
            for sub in subs:
                assert sub in class_locals, (
                    f"Domain '{domain}' required subsystem '{sub}' not found in taxonomy"
                )

    def test_predefined_domains_constant(self):
        assert PREDEFINED_DOMAINS == {"cubesat", "uam", "rocket", "lunar"}

    def test_classes_for_domain_returns_domain_subset(self):
        cubesat_classes = AEROSPACE_TAXONOMY.classes_for_domain("cubesat")
        assert all("cubesat" in c.domains for c in cubesat_classes)
        assert len(cubesat_classes) >= 1

    def test_get_class_returns_correct_def(self):
        cls = AEROSPACE_TAXONOMY.get_class("NanoSatellite")
        assert cls is not None
        assert cls.local == "NanoSatellite"
        assert cls.parent == "Satellite"

    def test_get_class_returns_none_for_unknown(self):
        assert AEROSPACE_TAXONOMY.get_class("NonExistentClass") is None

    def test_no_duplicate_data_property_locals(self):
        locals_list = [dp.local for dp in AEROSPACE_TAXONOMY.data_properties]
        assert len(locals_list) == len(set(locals_list))
