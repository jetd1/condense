# By Jet 2024
import torch


def sliding_window_inference(
        inputs, model, crop_size, stride, n_out_channels):
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    batch_size, _, h_img, w_img = inputs.size()
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = inputs.new_zeros((batch_size, n_out_channels, h_img, w_img))
    count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))

    patch_indices = [(h_idx * h_stride, w_idx * w_stride)
                     for h_idx in range(h_grids) for w_idx in range(w_grids)]
    patch_batch = []

    for y1, x1 in patch_indices:
        y2 = min(y1 + h_crop, h_img)
        x2 = min(x1 + w_crop, w_img)
        y1 = max(y2 - h_crop, 0)
        x1 = max(x2 - w_crop, 0)
        patch_batch.append(inputs[:, :, y1:y2, x1:x2])

    patch_batch = torch.cat(patch_batch, dim=0)
    patch_preds = model(patch_batch)

    for (y1, x1), patch_pred in zip(patch_indices, patch_preds.split(1, dim=0)):
        y2 = min(y1 + h_crop, h_img)
        x2 = min(x1 + w_crop, w_img)
        y1 = max(y2 - h_crop, 0)
        x1 = max(x2 - w_crop, 0)
        preds[:, :, y1:y2, x1:x2] += patch_pred.squeeze(0)
        count_mat[:, :, y1:y2, x1:x2] += 1

    seg_logits = torch.divide(preds, count_mat)
    return seg_logits
