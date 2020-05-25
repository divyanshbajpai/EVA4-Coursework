import visualize as viz
import torch
def validate(model, criterion, device, validate_loader):
    with torch.no_grad():
        for batchidx, data in enumerate(validate_loader):
            data["fg"] = data["fg"].to(device)
            data["bg"] = data["bg"].to(device)
            data["mask"] = data["mask"].to(device)
            output = model(data["fg"])
            break
    #           loss = criterion(output, data["mask"])
    #           return loss
    viz.show(output.detach().cpu(),data["mask"].detach().cpu())