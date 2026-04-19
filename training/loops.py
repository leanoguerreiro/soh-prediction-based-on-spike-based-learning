import logging
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
from tqdm import tqdm
from utils.common import add_gaussian_noise

logger = logging.getLogger("BatteryPipeline")

def train_standard(model, train_loader, X_val, y_val, X_test, config):
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss, patience_ctr, best_w, best_epoch = float('inf'), 0, None, 0
    history = {'train_loss': [], 'val_loss': []}

    epoch_bar = tqdm(range(config.epochs), desc="[Standard] Treinando", leave=False)
    for epoch in epoch_bar:
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(add_gaussian_noise(batch_x, std=config.noise_std)), batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = sum(batch_losses) / len(batch_losses)
        history['train_loss'].append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        history['val_loss'].append(val_loss)

        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", patience=patience_ctr)

        if val_loss < best_loss:
            best_loss, patience_ctr, best_w, best_epoch = val_loss, 0, model.state_dict(), epoch + 1
        else:
            patience_ctr += 1
            if patience_ctr >= config.patience:
                logger.info(f"      ↳ Early Stopping na época {epoch+1}. Melhor Val Loss: {best_loss:.4f} (Época {best_epoch})")
                break
    else:
        logger.info(f"      ↳ Treino concluído ({config.epochs} épocas). Melhor Val Loss: {best_loss:.4f} (Época {best_epoch})")

    model.load_state_dict(best_w)
    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy().flatten()
    return preds, history


def train_physics(model, train_loader, X_val, y_val, X_test, config, lambda_phys=0.2):
    criterion_mae = nn.HuberLoss(delta=1.0)
    criterion_mse = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss, patience_ctr, best_w, best_epoch = float('inf'), 0, None, 0
    history = {'train_loss': [], 'val_loss': []}

    epoch_bar = tqdm(range(config.epochs), desc="[Physics] Treinando", leave=False)
    for epoch in epoch_bar:
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            soh_pred, phys_pred = model(add_gaussian_noise(batch_x, std=config.noise_std))
            loss = criterion_mae(soh_pred, batch_y) + (lambda_phys * criterion_mse(phys_pred, batch_y))
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = sum(batch_losses) / len(batch_losses)
        history['train_loss'].append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = criterion_mae(model(X_val)[0], y_val).item()
        history['val_loss'].append(val_loss)

        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", patience=patience_ctr)

        if val_loss < best_loss:
            best_loss, patience_ctr, best_w, best_epoch = val_loss, 0, model.state_dict(), epoch + 1
        else:
            patience_ctr += 1
            if patience_ctr >= config.patience:
                logger.info(f"      ↳ Early Stopping na época {epoch+1}. Melhor Val Loss: {best_loss:.4f} (Época {best_epoch})")
                break
    else:
        logger.info(f"      ↳ Treino concluído ({config.epochs} épocas). Melhor Val Loss: {best_loss:.4f} (Época {best_epoch})")

    model.load_state_dict(best_w)
    model.eval()
    with torch.no_grad():
        preds = model(X_test)[0].cpu().numpy().flatten()
    return preds, history


def train_curriculum(model, train_loader, X_val, y_val, X_test, config):
    criterion_mae, criterion_mse = nn.HuberLoss(delta=1.0), nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss, patience_ctr, best_w, best_epoch = float('inf'), 0, None, 0
    history = {'train_loss': [], 'val_loss': [], 'lambda': []}

    epoch_bar = tqdm(range(config.epochs), desc="[Curriculum] Treinando", leave=False)
    for epoch in epoch_bar:
        model.set_epoch(epoch)
        lam = model.get_lambda()
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            soh_pred, phys_pred = model(add_gaussian_noise(batch_x, std=config.noise_std))
            loss = criterion_mae(soh_pred, batch_y) + lam * criterion_mse(phys_pred, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = sum(batch_losses) / len(batch_losses)
        history['train_loss'].append(train_loss)
        history['lambda'].append(lam)

        model.eval()
        with torch.no_grad():
            val_loss = criterion_mae(model(X_val)[0], y_val).item()
        history['val_loss'].append(val_loss)

        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", lam=f"{lam:.2f}", patience=patience_ctr)

        if val_loss < best_loss:
            best_loss, patience_ctr, best_w, best_epoch = val_loss, 0, {k: v.clone() for k, v in model.state_dict().items()}, epoch + 1
        else:
            patience_ctr += 1
            if patience_ctr >= config.patience:
                logger.info(f"      ↳ Early Stopping na época {epoch+1}. Melhor Val Loss: {best_loss:.4f} (Época {best_epoch})")
                break
    else:
        logger.info(f"      ↳ Treino concluído ({config.epochs} épocas). Melhor Val Loss: {best_loss:.4f} (Época {best_epoch})")

    model.load_state_dict(best_w)
    model.eval()
    with torch.no_grad():
        preds = model(X_test)[0].cpu().numpy().flatten()
    return preds, history


def train_spikingjelly(model, train_loader, X_val, y_val, X_test, config):
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss, patience_ctr, best_w, best_epoch = float('inf'), 0, None, 0
    history = {'train_loss': [], 'val_loss': []}

    epoch_bar = tqdm(range(config.epochs), desc="[SNN] Treinando", leave=False)
    for epoch in epoch_bar:
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(add_gaussian_noise(batch_x, std=config.noise_std)), batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            functional.reset_net(model)
            batch_losses.append(loss.item())

        train_loss = sum(batch_losses) / len(batch_losses)
        history['train_loss'].append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
            functional.reset_net(model)
        history['val_loss'].append(val_loss)

        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", patience=patience_ctr)

        if val_loss < best_loss:
            best_loss, patience_ctr, best_w, best_epoch = val_loss, 0, model.state_dict(), epoch + 1
        else:
            patience_ctr += 1
            if patience_ctr >= config.patience:
                logger.info(f"      ↳ Early Stopping na época {epoch+1}. Melhor Val Loss: {best_loss:.4f} (Época {best_epoch})")
                break
    else:
        logger.info(f"      ↳ Treino concluído ({config.epochs} épocas). Melhor Val Loss: {best_loss:.4f} (Época {best_epoch})")

    model.load_state_dict(best_w)
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        functional.reset_net(model)
    return preds.cpu().numpy().flatten(), history