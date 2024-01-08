# PorP

``` python

    import utils as ut

    DOMAINS = {
        0: ["a", "b", "c"],
        1: ["d", "e", "f"],
        2: ["g", "h", "i"],
        3: ["j", "k", "l"],
    }

    cards = ut.generate_card_list_from_domain_dict(
        DOMAINS,
        n_cards = 100, n_symbols_per_card = 5
    )
```

