import json
import unittest

from fastapi.testclient import TestClient
from main import predict_credit, app
from sample_call import test_dict

client = TestClient(app)


class TestAPI(unittest.TestCase):
    def test_if_api_call_request_is_correct(self):
        # given
        request = test_dict.copy()

        # when
        api_result = json.loads(predict_credit(request))

        # then
        api_result_expected_prediction = "crédit accordé"
        self.assertEqual(api_result_expected_prediction, api_result['prediction'])

    def test_if_bad_api_call_request_returns_correct_status_code(self):
        # given
        request = {'toto': 2, 'tata': 4}

        # when
        api_response = client.post(
            "/",
            json=request,
        )

        # then
        api_response_status_code_expected = 400
        self.assertEqual(api_response_status_code_expected, api_response.status_code)
