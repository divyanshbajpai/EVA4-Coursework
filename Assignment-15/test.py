import visualize as viz
def test(model, criterion, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0

  with torch.no_grad():
    for data in test_loader:
      data["fg"] = data["fg"].to(device)
      data["bg"] = data["bg"].to(device)
      data["mask"] = data["mask"].to(device)
      output = model(data["fg"])

      test_loss += criterion(output, data["mask"], reduction='sum' ).item()
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()

      show(output.cpu(), nrow=2)
    test_loss /= len(test_loader.dataset)
    # viz.show(output.detach().cpu(), nrow=4)