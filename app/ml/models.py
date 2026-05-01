import torch
import torch.nn as nn
import torch.nn.functional as F


class RecipeTwoTower(nn.Module):
    """Two-tower neural net for recipe recommendation.

    User tower: ID embedding + SBERT history vector.
    Recipe tower: ID embedding + SBERT text vector.
    """

    def __init__(self, num_users, num_recipes, text_dim, embed_dim=128):
        """Args:
            num_users: Total number of users in the training set.
            num_recipes: Total number of recipes in the training set.
            text_dim: Dimension of the SBERT text embeddings.
            embed_dim: Output embedding dimension for both towers.
        """
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.user_fc = nn.Linear(embed_dim, embed_dim)
        self.history_fc = nn.Linear(text_dim, embed_dim)
        self.user_out_fc = nn.Linear(embed_dim * 2, embed_dim)
        self.recipe_embedding = nn.Embedding(num_recipes, embed_dim)
        self.text_fc = nn.Linear(text_dim, embed_dim)
        self.item_fc = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(p=0.2)

    def _user_vec(self, user_idx, hist_vec):
        """Compute a raw (non-normalized) user embedding.

        Used in forward() during training; the training loop normalizes externally.

        Args:
            user_idx: Tensor of user indices.
            hist_vec: SBERT history vector tensor.

        Returns:
            Unnormalized combined user embedding tensor.
        """
        u = F.relu(self.user_fc(self.user_embedding(user_idx)))
        h = F.relu(self.history_fc(hist_vec))
        return F.relu(self.user_out_fc(torch.cat([u, h], dim=1)))

    def _item_vec(self, recipe_idx, text_features):
        """Compute a raw (non-normalized) recipe embedding.

        Used in forward() during training; the training loop normalizes externally.

        Args:
            recipe_idx: Tensor of recipe indices.
            text_features: SBERT text vector tensor.

        Returns:
            Unnormalized combined recipe embedding tensor.
        """
        r = self.recipe_embedding(recipe_idx)
        t = F.relu(self.text_fc(text_features))
        return F.relu(self.item_fc(torch.cat([r, t], dim=1)))

    def forward(self, user_idx, recipe_idx, text_features, hist_vec):
        """Run a forward pass and return dot-product scores.

        Args:
            user_idx: Tensor of user indices.
            recipe_idx: Tensor of recipe indices.
            text_features: SBERT text vector tensor.
            hist_vec: SBERT history vector tensor.

        Returns:
            1D tensor of scores, one per sample in the batch.
        """
        u_vec = self.dropout(self._user_vec(user_idx, hist_vec))
        i_vec = self.dropout(self._item_vec(recipe_idx, text_features))
        return (u_vec * i_vec).sum(dim=1)

    def get_user_vec(self, user_idx, hist_vec):
        """Return an L2-normalized user embedding for retrieval.

        The dot product with a normalized item vec equals cosine similarity.

        Args:
            user_idx: Tensor of user indices.
            hist_vec: SBERT history vector tensor.

        Returns:
            L2-normalized user embedding tensor.
        """
        u = F.relu(self.user_fc(self.user_embedding(user_idx)))
        h = F.relu(self.history_fc(hist_vec))
        return F.normalize(F.relu(self.user_out_fc(torch.cat([u, h], dim=1))), dim=1)

    def get_item_vec(self, recipe_idx, text_features):
        """Return an L2-normalized recipe embedding for retrieval.

        The dot product with a normalized user vec equals cosine similarity.

        Args:
            recipe_idx: Tensor of recipe indices.
            text_features: SBERT text vector tensor.

        Returns:
            L2-normalized recipe embedding tensor.
        """
        r = self.recipe_embedding(recipe_idx)
        t = F.relu(self.text_fc(text_features))
        return F.normalize(F.relu(self.item_fc(torch.cat([r, t], dim=1))), dim=1)