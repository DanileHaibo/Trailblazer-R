"""Original native scalar codec and action-stream validation."""
import math

START, END, EOS = 10004, 15004, 8710
LOW = (-.9375, -.9375, -.9375, -.32571429, -.375, -.375, -1.)
HIGH = (.9375, .9375, .9375, .375, .375, .375, 1.)
BINS = tuple(-1 + 2 * i / 255 for i in range(256))

class NativeProtocolError(RuntimeError):
    pass

def decode_step(tokens):
    if len(tokens) != 9 or tokens[0] != START or tokens[-1] != END:
        raise ValueError('malformed native scalar step')
    if any(type(token) is not int or not START+1 <= token <= START+257 for token in tokens[1:8]):
        raise ValueError('non-native scalar token')
    normalized = [(BINS[i]+BINS[i+1])/2 for i in
                  [max(0, min(254, token-START-2)) for token in tokens[1:8]]]
    return [(x+1)/2*(hi-lo+1e-8)+lo for x, lo, hi in zip(normalized, LOW, HIGH)]

def valid(tokens, horizon):
    blocks = []
    current = None
    for index, token in enumerate(tokens):
        if token == START:
            current = index
        elif token == END and current is not None:
            block = tokens[current:index+1]
            if len(block) != 9:
                raise NativeProtocolError(f'native malformed action length {len(block)}')
            blocks.append(block)
            current = None
    if not 1 <= len(blocks) <= horizon or sum(blocks, [])+[EOS] != tokens:
        raise NativeProtocolError('noncanonical action stream')
    for block in blocks:
        decode_step(block)
    return blocks
