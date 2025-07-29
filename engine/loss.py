"""___Modules_______________________________________________________________"""

# Python
import numpy as np

"""___Classes_______________________________________________________________"""

class Loss():

    def forward(self,
                y_pred: np.array,
                y_true: np.array,
                loss_function: str
                ) -> None:
        if type(loss_function) == str:
            if loss_function == 'CCE':
                self.CCE(y_pred, y_true)
            elif loss_function == 'MSE':
                self.MSE(y_pred, y_true)
            elif loss_function == 'MAE':
                self.MAE(y_pred, y_true)
        else:
            self.output = loss_function(y_pred, y_true)

    def CCE(self,
            y_pred: np.array,
            y_true: np.array
            ) -> None:
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
            uncorrect_confidences = 1 + np.sum(y_pred_clipped, axis=1) - correct_confidences

        if len(y_true.shape) == 2:
            raise ValueError("CCE function not updated")
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        loss = -np.log(correct_confidences) + np.log(uncorrect_confidences)     # Forme abrégée
        data_loss = np.mean(loss)

        self.output = data_loss

    def MSE(self,
            y_pred: np.array,
            y_true: np.array
            ) -> None:
        samples = len(y_pred)
        Squared_errors = np.square(y_true - y_pred)
        Mean_squared_error = np.sum(Squared_errors) / samples
        self.output = Mean_squared_error

    def MAE(self,
            y_pred: np.array,
            y_true: np.array
            ) -> None:
        samples = len(y_true)
        Absolute_errors = np.abs(y_true - y_pred)
        Mean_absolute_error = np.sum(Absolute_errors) / samples
        self.output = Mean_absolute_error
