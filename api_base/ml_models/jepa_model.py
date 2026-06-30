"""
JEPA: Joint Embedding Predictive Architecture for Tabular Data
===============================================================
Dựa trên paper I-JEPA (Assran et al., CVPR 2023) và LeCun (2022).

Nguyên lý:
  - Encoder biến input thành embedding (vector đặc trưng)
  - Target Encoder giống Encoder nhưng cập nhật bằng EMA (trung bình động)
  - Predictor dự đoán embedding của target từ embedding của context
  - Học tự giám sát (self-supervised): KHÔNG cần nhãn
  - Loss = MSE giữa predicted embedding và target embedding
  - Stop-gradient trên Target Encoder (ngăn collapse)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

logger = logging.getLogger(__name__)


class EncoderMLP(nn.Module):
    """Bộ mã hóa: input features → embedding vector"""

    def __init__(self, input_dim: int, hidden_dims: list, embedding_dim: int = 32):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
            ])
            prev = h
        layers.append(nn.Linear(prev, embedding_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PredictorMLP(nn.Module):
    """Bộ dự đoán: context_embedding → target_embedding"""

    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


class JEPA:
    """
    Joint Embedding Predictive Architecture

    Components:
      - encoder: input → embedding (có gradient)
      - target_encoder: input → embedding (không gradient, EMA update)
      - predictor: context_embedding → target_embedding

    Training:
      1. Tạo context (mask features) và target (original)
      2. Encode context → ctx_emb (encoder)
      3. Encode target → tgt_emb (target_encoder, stop-gradient)
      4. Predict: pred_emb = predictor(ctx_emb)
      5. Loss = MSE(pred_emb, tgt_emb.detach())
      6. Update target_encoder: θ_tgt = τ * θ_tgt + (1-τ) * θ_enc
    """

    def __init__(self,
                 input_dim: int,
                 embedding_dim: int = 32,
                 hidden_dims: list = None,
                 predictor_hidden: int = 64,
                 device: str = 'cpu'):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.device = device

        # Tạo encoder và target_encoder (cùng cấu trúc, khác tham số)
        self.encoder = EncoderMLP(input_dim, self.hidden_dims, embedding_dim).to(device)
        self.target_encoder = EncoderMLP(input_dim, self.hidden_dims, embedding_dim).to(device)

        # Khởi tạo target_encoder giống hệt encoder
        self.target_encoder.load_state_dict(self.encoder.state_dict())

        # Freeze target_encoder (không train trực tiếp)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor
        self.predictor = PredictorMLP(embedding_dim, predictor_hidden).to(device)

        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=0.001
        )

        self._trained = False

    def _momentum_update(self, tau: float = 0.995):
        """Cập nhật target_encoder bằng EMA (trung bình động)"""
        with torch.no_grad():
            for p_enc, p_tgt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                p_tgt.data = tau * p_tgt.data + (1 - tau) * p_enc.data

    def _create_context_target(self, x: torch.Tensor, mask_rate: float = 0.3):
        """
        Tạo context (bị mask) và target (gốc) cho JEPA.
        Context: một số features bị zero-out
        Target: features gốc
        """
        # Context: random mask features
        mask = torch.rand(x.shape[0], self.input_dim, device=self.device) > mask_rate
        context = x * mask.float()

        # Target: gốc (hoặc thêm noise nhẹ)
        target = x

        return context, target

    def train(self, X: np.ndarray,
              epochs: int = 200,
              batch_size: int = 32,
              mask_rate: float = 0.3,
              lr: float = 0.001,
              verbose: bool = True):
        """Train JEPA self-supervised (không cần nhãn y)"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        bs = max(2, min(batch_size, len(X) // 2))
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

        for p in self.optimizer.param_groups:
            p['lr'] = lr

        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0

            for (batch_x,) in loader:
                context, target = self._create_context_target(batch_x, mask_rate)

                # Encode context (có gradient)
                ctx_emb = self.encoder(context)

                # Encode target (không gradient - target_encoder)
                with torch.no_grad():
                    tgt_emb = self.target_encoder(target)

                # Dự đoán target embedding từ context embedding
                pred_emb = self.predictor(ctx_emb)

                # Loss = MSE (chuẩn hóa embedding trước)
                loss = nn.functional.mse_loss(
                    nn.functional.normalize(pred_emb, dim=1),
                    nn.functional.normalize(tgt_emb, dim=1)
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Cập nhật target_encoder bằng EMA
                self._momentum_update(tau=0.995)

                total_loss += loss.item()
                n_batches += 1

            if verbose and (epoch + 1) % 50 == 0:
                logger.info(f"  JEPA epoch {epoch+1}/{epochs}, loss={total_loss/n_batches:.4f}")

        self._trained = True
        if verbose:
            logger.info(f"  JEPA training done: {epochs} epochs")

    def train_supervised(self, X: np.ndarray, y: np.ndarray,
                          epochs: int = 50, lr: float = 0.0005, verbose: bool = True):
        """
        Supervised fine-tuning: dùng labels để shaping embedding space.
        Thêm classification head + contrastive loss.
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        bs = max(2, min(32, len(X)))
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

        n_classes = int(y.max()) + 1
        classifier = nn.Linear(self.embedding_dim, n_classes).to(self.device)

        params = list(self.encoder.parameters()) + list(classifier.parameters())
        optimizer = optim.Adam(params, lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                emb = self.encoder(batch_x)
                logits = classifier(emb)

                # Cross-entropy loss (chính)
                ce_loss = nn.functional.cross_entropy(logits, batch_y)

                # Contrastive regularization: kéo embedding cùng lớp lại gần
                emb_norm = nn.functional.normalize(emb, dim=1)
                sim_matrix = emb_norm @ emb_norm.T
                pos_mask = batch_y.unsqueeze(0) == batch_y.unsqueeze(1)
                pos_mask = pos_mask.float()
                pos_sim = (sim_matrix * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
                neg_sim = (sim_matrix * (1 - pos_mask)).sum(dim=1) / (1 - pos_mask).sum(dim=1).clamp(min=1)
                contrast_loss = -(pos_sim - neg_sim).mean()

                loss = ce_loss + 0.1 * contrast_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(f"  Supervised epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.4f}")

        if verbose:
            logger.info(f"  Supervised fine-tuning done")

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Lấy embedding từ encoder (dùng cho SVM)"""
        if not self._trained:
            raise ValueError("JEPA chưa được train")
        self.encoder.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            embeddings = self.encoder(X_tensor).cpu().numpy()
        return embeddings

    def save(self, path: str):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'target_encoder': self.target_encoder.state_dict(),
            'predictor': self.predictor.state_dict(),
            'input_dim': self.input_dim,
            'embedding_dim': self.embedding_dim,
        }, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(state['encoder'])
        self.target_encoder.load_state_dict(state['target_encoder'])
        self.predictor.load_state_dict(state['predictor'])
        self._trained = True
