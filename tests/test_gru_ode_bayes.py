import unittest
import os
import torch
from flood_forecast.gru_ode_bayes.gru_ode_bayes_model import GRU_ODE_Bayes_Classifier

class TestGRU_ODE_Bayes_Classifier(unittest.TestCase):
    def setUp(self):
        self.preprocessed_data = self.preprocessed_data = make_data(os.path.join(
            os.path.dirname(__file__), "test_init", "TODO_FILENAME.csv"), ["cfs"], 72)

    def test_train_model(self):
        with tempfile.TemporaryDirectory() as param_directory:
            gobc_model = GRU_ODE_Bayes_Classifier(self.preprocessed_data, 1, 64,
                                        param_output_path=param_directory)
            config = gobc_model.config_writer(defaults=True, to_file=False)
            gobc_model.train(self.preprocessed_data,
                                        config, n_epochs=1, tensorboard=True)
            self.assertTrue(model)

    def test_tf_data(self):
        dirname = os.path.dirname(__file__)
        # Test that Tensorboard directory was indeed created
        self.assertTrue(os.listdir(os.path.join(dirname)))

    def test_create_model(self):
        with tempfile.TemporaryDirectory() as param_directory:
            gobc_model = GRU_ODE_Bayes_Classifier(self.preprocessed_data, 1, 64,
                                        param_output_path=param_directory)
            self.assertNotEqual(gobc_model.config.batch_size, 20)
            self.assertIsNotNone(gobc_model)

    def test_resume_ckpt(self):
        """ This test is dependent on test_train_model succeding"""
            gobc_model = GRU_ODE_Bayes_Classifier(self.preprocessed_data, 1, 64,
                                        param_output_path=param_directory)
        with tempfile.TemporaryDirectory() as checkpoint:
            torch.save(gobc_model.encoder.state_dict(), os.path.join(checkpoint, "encoder.pth"))
            torch.save(gobc_model.decoder.state_dict(), os.path.join(checkpoint, "decoder.pth"))
            gobc_model = GRU_ODE_Bayes_Classifier(self.preprocessed_data, 1, 64,
                                        param_output_path=param_directory)
            gobc_model.train(TODO_traindata)
            gobc_model.save_model(checkpoint)

            self.assertTrue(gobc_model)

if __name__ == '__main__':
    unittest.main()