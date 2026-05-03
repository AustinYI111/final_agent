from py_mini_racer import MiniRacer
ctx = MiniRacer()
result = ctx.eval('({a: 1, b: "hello", c: [1,2,3]})')
print('type:', type(result))
print('dir:', [x for x in dir(result) if not x.startswith('_')])
try:
    print('dict:', dict(result))
except Exception as e:
    print('dict error:', e)
try:
    print('keys:', list(result.keys()))
except Exception as e:
    print('keys error:', e)
