use crate::consts;

pub fn factorial_exact(n: u64) -> u64 { (1..=n).product() }

pub fn factorial(n: u64) -> u64 {
    if n > 11 {
        factorial_exact(n)
    } else {
        consts::FACTORIALS_16[n as usize]
    }
}

pub fn log_factorial_stirling(n: u64) -> f64 {
    if n > 254 {
        let x = (n + 1) as f64;

        (x - 0.5) * x.ln() - x + 0.5 * (2.0 * consts::PI).ln() + 1.0 / (12.0 * x)
    } else {
        consts::LOG_FACTORIALS_255[n as usize]
    }
}

pub fn choose(n: u64, k: u64) -> u64 {
    let k = if k > n - k { n - k } else { k };

    (0..k).fold(1, |acc, i| acc * (n - i) / (i + 1))
}

#[cfg(feature = "serde")]
pub(crate) mod serde_arrays {
    use std::{convert::TryInto, marker::PhantomData};

    use serde_crate::{
        de::{SeqAccess, Visitor},
        ser::SerializeTuple,
        Deserialize, Deserializer, Serialize, Serializer,
    };
    pub fn serialize<S: Serializer, T: Serialize, const N: usize>(
        data: &[T; N],
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        let mut s = ser.serialize_tuple(N)?;

        for item in data {
            s.serialize_element(item)?;
        }

        s.end()
    }

    struct ArrayVisitor<T, const N: usize>(PhantomData<T>);

    impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<T, N>
    where
        T: Deserialize<'de>,
    {
        type Value = [T; N];

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str(&format!("an array of length {}", N))
        }

        #[inline]
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut data = Vec::with_capacity(N);

            for _ in 0..N {
                match (seq.next_element())? {
                    Some(val) => data.push(val),
                    None => return Err(serde_crate::de::Error::invalid_length(N, &self)),
                }
            }

            match data.try_into() {
                Ok(arr) => Ok(arr),
                Err(_) => unreachable!(),
            }
        }
    }
    pub fn deserialize<'de, D, T, const N: usize>(deserializer: D) -> Result<[T; N], D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        deserializer.deserialize_tuple(N, ArrayVisitor::<T, N>(PhantomData))
    }
}
