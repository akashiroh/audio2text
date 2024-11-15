
def batch_collate_fn(batch):
    """pad inputs and outputs with zeros"""

    inputs, outputs = zip(*batch)

    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=-1)
    outputs = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True, padding_value=-1)

    return inputs, outputs
