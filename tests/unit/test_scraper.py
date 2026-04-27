"""Unit tests for the RSS scraper speaker extraction."""

import unittest

from src.scraper import _extract_speakers


class _FakeContent:
    def __init__(self, html):
        self.content = [{"value": html}] if html else []


class TestExtractSpeakers(unittest.TestCase):

    def test_standard_hosts_and_guests(self):
        html = (
            "<p>No <strong>NerdCast</strong> de hoje, <strong>Alottoni</strong> "
            "e <strong>Azaghal</strong> recebem <strong>Altay</strong> "
            "<strong>de</strong> <strong>Souza</strong>, <strong>Ana</strong> "
            "<strong>Arantes</strong> e <strong>Sr. K</strong> para um papo.</p>"
        )
        result = _extract_speakers(_FakeContent(html))
        self.assertEqual(result, "Alottoni|Azaghal|Altay de Souza|Ana Arantes|Sr. K")

    def test_parenthesized_alias_excluded(self):
        html = (
            "<p>Junte-se a <strong>Alottoni</strong>, <strong>Pedro</strong> "
            "<strong>Pallotta</strong> (<strong>Space</strong> <strong>Orbit</strong>), "
            "<strong>Katiucha</strong> <strong>Barcelos</strong> e "
            "<strong>Azaghal</strong> numa viagem.</p>"
        )
        result = _extract_speakers(_FakeContent(html))
        self.assertEqual(result, "Alottoni|Pedro Pallotta|Katiucha Barcelos|Azaghal")

    def test_nerdcast_filtered_out(self):
        html = "<p>No <strong>NerdCast</strong> de hoje, <strong>Alottoni</strong> e <strong>Azaghal</strong>.</p>"
        result = _extract_speakers(_FakeContent(html))
        self.assertEqual(result, "Alottoni|Azaghal")

    def test_no_content(self):
        result = _extract_speakers(_FakeContent(None))
        self.assertEqual(result, "")

    def test_empty_html(self):
        result = _extract_speakers(_FakeContent(""))
        self.assertEqual(result, "")

    def test_no_bold_tags(self):
        html = "<p>Um episódio sem nomes em negrito.</p>"
        result = _extract_speakers(_FakeContent(html))
        self.assertEqual(result, "")

    def test_single_speaker(self):
        html = "<p><strong>Alottoni</strong> fala sobre coisas.</p>"
        result = _extract_speakers(_FakeContent(html))
        self.assertEqual(result, "Alottoni")

    def test_only_second_paragraph_ignored(self):
        html = (
            "<p><strong>Alottoni</strong> e <strong>Azaghal</strong>.</p>"
            "<p><strong>Sociedade</strong> <strong>da</strong> <strong>Virtude</strong>.</p>"
        )
        result = _extract_speakers(_FakeContent(html))
        self.assertEqual(result, "Alottoni|Azaghal")

    def test_speakers_in_second_paragraph(self):
        html = (
            '<p>Lambda lambda lambda, nerds! Como diria um sábio.</p>'
            '<p>No <strong>NerdCast</strong> de hoje, <strong>Alottoni</strong> '
            'e <strong>Azaghal</strong> recebem <strong>Sr. K</strong> para um papo.</p>'
        )
        result = _extract_speakers(_FakeContent(html))
        self.assertEqual(result, "Alottoni|Azaghal|Sr. K")

    def test_first_p_only_nerdcast_skips_to_next(self):
        html = (
            '<p>No <strong>NerdCast</strong> de hoje.</p>'
            '<p><strong>Alottoni</strong> e <strong>Azaghal</strong>.</p>'
        )
        result = _extract_speakers(_FakeContent(html))
        self.assertEqual(result, "Alottoni|Azaghal")
