import visualize as viz
def train(model, criterion, device, train_loader, optimizer, epoch,scheduler,writer):
    # print("Start Model Training For Epoch ", str(epoch))
    model.train()
    epoch_loss = 0
    # print("Super call ended, starting loop")
    for batch_idx, data in enumerate(train_loader):
        data["fg"] = data["fg"].to(device)
        data["bg"] = data["bg"].to(device)
        data["mask"] = data["mask"].to(device)
        optimizer.zero_grad()
        output = model(data["fg"])
        loss = criterion(output, data["mask"])
        epoch_loss = epoch_loss + loss
        loss.backward()
        optimizer.step()
    scheduler.step(epoch_loss)
    writer.add_scalar('Epoch training loss', epoch_loss, epoch)
    #writer.add_scalar('Epoch training loss', epoch_loss, epoch)
    if epoch % 1 == 0 :
        viz.show(output.detach().cpu(),data["mask"].detach().cpu(),epoch) 